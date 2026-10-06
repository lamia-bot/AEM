# -*- mode: python ; coding: utf-8 -*-
"""Windows onedir build for MEA Analyzer Build 22.

The app intentionally uses an onedir layout rather than PyInstaller onefile:
there is no self-extraction step at launch, startup is more predictable, and
heavy scientific DLLs stay as ordinary files beside the executable.
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

project_root = Path(SPECPATH)

# Several packages are imported lazily/background-preloaded by the application.
# Explicitly list them so PyInstaller cannot omit them simply because import
# timing was optimized for startup responsiveness.
hiddenimports = [
    "scipy.signal",
    "pyarrow",
    "pyarrow.parquet",
    "pyarrow.csv",
    "pyarrow.compute",
    "xlsxwriter",
    "reportlab",
    "reportlab.platypus",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]
hiddenimports += collect_submodules("pyarrow")

# PyArrow ships native DLLs that must travel with the onedir application.
binaries = collect_dynamic_libs("pyarrow")

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pandas", "tkinter", "notebook", "jupyter"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MEA_Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    version=str(project_root / "packaging" / "windows_version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MEA_Analyzer",
)
