from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

CACHE_WARNING_BYTES = 10 * 1024**3


def cache_root() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        root = base / "MEA Analyzer" / "Cache"
    elif system == "Darwin":
        root = Path.home() / "Library" / "Caches" / "MEA Analyzer"
    else:
        root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "MEA_Analyzer"
    root.mkdir(parents=True, exist_ok=True)
    return root


def human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _entry_key(path: Path) -> str:
    fp = _fingerprint(path)
    raw = f"{fp['resolved_path']}|{fp['size']}|{fp['mtime_ns']}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


@dataclass
class CacheEntry:
    key: str
    directory: Path
    source_name: str
    source_path: str
    last_used: float
    size_bytes: int

    @property
    def last_used_text(self) -> str:
        return datetime.fromtimestamp(self.last_used).strftime("%Y-%m-%d %H:%M")


class CacheManager:
    def __init__(self) -> None:
        self.root = cache_root()

    def entry_dir(self, source_path: Path) -> Path:
        return self.root / _entry_key(source_path)

    def valid_entry(self, source_path: Path) -> Path | None:
        directory = self.entry_dir(source_path)
        meta_path = directory / "metadata.json"
        parquet_path = directory / "recording.parquet"
        preview_path = directory / "preview.npz"
        if not (meta_path.exists() and preview_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_fingerprint") != _fingerprint(source_path):
                return None
            # Build 19 CSV caches are intentionally preview-only until the user
            # chooses an electrode. XLSX and older CSV caches still require the
            # common Parquet cache.
            lazy_csv = (
                str(meta.get("source_format", "")).casefold() == "csv"
                and str(meta.get("cache_mode", "")).casefold() == "csv_lazy"
            )
            if not lazy_csv and not parquet_path.exists():
                return None
        except Exception:
            return None
        return directory

    def touch(self, directory: Path) -> None:
        meta_path = directory / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["last_used"] = time.time()
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def entries(self) -> list[CacheEntry]:
        rows: list[CacheEntry] = []
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            meta_path = directory / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append(
                    CacheEntry(
                        key=directory.name,
                        directory=directory,
                        source_name=meta.get("source_name", "Unknown"),
                        source_path=meta.get("source_path", ""),
                        last_used=float(meta.get("last_used", 0)),
                        size_bytes=_dir_size(directory),
                    )
                )
            except Exception:
                continue
        return sorted(rows, key=lambda x: x.last_used)

    def total_size(self) -> int:
        return sum(entry.size_bytes for entry in self.entries())

    def over_warning_limit(self) -> bool:
        return self.total_size() > CACHE_WARNING_BYTES

    def clear_entries(self, directories: Iterable[Path]) -> None:
        for directory in directories:
            try:
                shutil.rmtree(directory)
            except FileNotFoundError:
                pass

    def clear_all(self) -> None:
        self.clear_entries(entry.directory for entry in self.entries())
