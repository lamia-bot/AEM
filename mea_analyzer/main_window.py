from __future__ import annotations

import sys
import traceback
import logging
from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QButtonGroup,
    QFrame,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .cache import CACHE_WARNING_BYTES, CacheManager, human_size
from .analysis_logic import (
    apply_v7_smoothing,
    calculate_z_score_v7,
    detect_spikes_v7,
    divide_spikes_into_blocks_v7,
    compute_isi_results_v7,
    extract_snippets_v7,
    compute_selected_mean_v7,
    determine_baseline_v7,
    extract_landmarks_v7,
    compute_results_v7,
)
from .io_loader import (
    detect_recording_format,
    fast_electrode_cache_ready,
    load_electrode_range,
    load_range_preview,
    prepare_csv_cache,
    prepare_excel_cache,
    prepare_fast_electrode_cache,
    prepare_range_preview,
)
from .state import ANALYSIS_STEPS, AnalysisState, StepState
from .widgets import BaselinePlot, DetectionPlot, ElectrodeCard, LandmarksPlot, LargeSignalPreview, RangeSelectionPlot, SnippetAveragingPlot

WINDOW_SCREEN_RATIO = 0.56
# Build 11: keep the same height, but make the window ~10% wider.
# The navigation and control panels stay fixed-width, so the recovered width
# goes to the analysis plots.
MIN_APP_WIDTH = 902
MIN_APP_HEIGHT = 560
MAX_APP_WIDTH = 1056
MAX_APP_HEIGHT = 640
NAV_WIDTH = 185


class RecordingLoadWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, cache_manager: CacheManager):
        super().__init__()
        self.path = path
        self.cache_manager = cache_manager

    def run(self):
        try:
            source_format = detect_recording_format(self.path)
            self.progress.emit(1, f"Detected {source_format.upper()} recording")
            loader = prepare_csv_cache if source_format == "csv" else prepare_excel_cache
            result = loader(
                self.path,
                self.cache_manager,
                lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(result)
        except Exception as exc:
            logging.getLogger(__name__).exception("Recording preparation failed")
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=3)}")


class FastElectrodeCacheWorker(QObject):
    """Build a memory-mapped Time + selected-electrode sidecar off the GUI thread."""

    progress = Signal(str, int)
    finished = Signal(str)
    failed = Signal(str, str)

    def __init__(self, recording, electrode: str):
        super().__init__()
        self.recording = recording
        self.electrode = electrode

    def run(self):
        try:
            prepare_fast_electrode_cache(
                self.recording,
                self.electrode,
                lambda pct: self.progress.emit(self.electrode, int(pct)),
            )
            self.finished.emit(self.electrode)
        except Exception as exc:
            logging.getLogger(__name__).exception("Fast electrode cache preparation failed for %s", self.electrode)
            self.failed.emit(self.electrode, f"{exc}\n\n{traceback.format_exc(limit=3)}")


class RangePreviewWorker(QObject):
    """Prepare the enhanced ~30k-point selected-electrode Range preview."""

    progress = Signal(str, int)
    finished = Signal(str)
    failed = Signal(str, str)

    def __init__(self, recording, electrode: str):
        super().__init__()
        self.recording = recording
        self.electrode = electrode

    def run(self):
        try:
            prepare_range_preview(
                self.recording,
                self.electrode,
                lambda pct: self.progress.emit(self.electrode, int(pct)),
            )
            self.finished.emit(self.electrode)
        except Exception as exc:
            logging.getLogger(__name__).exception("Range preview preparation failed for %s", self.electrode)
            self.failed.emit(self.electrode, f"{exc}\n\n{traceback.format_exc(limit=3)}")


class RangeLoadWorker(QObject):
    """Load the exact selected interval at full resolution off the GUI thread."""

    progress = Signal(int)
    finished = Signal(object, object, float, float, str)
    failed = Signal(str)

    def __init__(self, recording, electrode: str, start: float, end: float):
        super().__init__()
        self.recording = recording
        self.electrode = electrode
        self.start = float(start)
        self.end = float(end)

    def run(self):
        try:
            time_values, voltage_values = load_electrode_range(
                self.recording,
                self.electrode,
                self.start,
                self.end,
                progress=lambda pct: self.progress.emit(int(pct)),
            )
            self.finished.emit(
                time_values, voltage_values, self.start, self.end, self.electrode
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Selected range loading failed")
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=3)}")


class BackgroundPreloadWorker(QObject):
    """Warm optional heavy modules after recording preparation has finished.

    Imports happen once per process and stay resident. Running them after the
    recording/cache stage avoids competing with first-open file preparation while
    reducing the chance of a delay when Detection or Save Results is reached.
    """
    finished = Signal()

    def run(self):
        try:
            from scipy.signal import savgol_filter  # noqa: F401
            from . import exporter  # noqa: F401
        except Exception:
            # Preloading is only an optimization; normal lazy imports remain the fallback.
            logging.getLogger(__name__).exception("Background dependency preload failed")
        finally:
            self.finished.emit()


class NavButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setMinimumHeight(38)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton {text-align:left; padding:8px 12px; border:none; border-radius:7px;"
            " font-size:13px; color:#1f2937;}"
            "QPushButton:hover {background:#eef2ff;}"
            "QPushButton:checked {background:#dbeafe; color:#1d4ed8; font-weight:700;}"
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MEA Analyzer v8 — Build 22")
        self._set_initial_window_size()
        self.state = AnalysisState()
        self.cache_manager = CacheManager()
        self.nav_buttons: dict[str, NavButton] = {}
        self.pages: dict[str, QWidget] = {}
        self.electrode_cards: dict[str, ElectrodeCard] = {}
        self.selected_preview_electrode: str | None = None
        self._worker_thread: QThread | None = None
        self._worker: RecordingLoadWorker | None = None
        self._preload_thread: QThread | None = None
        self._preload_worker: BackgroundPreloadWorker | None = None
        self._fast_electrode_thread: QThread | None = None
        self._fast_electrode_worker: FastElectrodeCacheWorker | None = None
        self._fast_electrode_name: str | None = None
        self._fast_electrode_requested: str | None = None
        self._fast_electrode_progress = 0
        self._fast_electrode_failed: set[str] = set()
        self._range_preview_thread: QThread | None = None
        self._range_preview_worker: RangePreviewWorker | None = None
        self._range_preview_electrode: str | None = None
        self._range_preview_requested: str | None = None
        self._range_preview_progress = 0
        self._range_preview_failed: set[str] = set()
        self._range_load_thread: QThread | None = None
        self._range_load_worker: RangeLoadWorker | None = None
        self._range_load_progress = 0
        self._range_continue_after_apply = False
        self._landmark_fullscreen_active = False
        self._landmark_was_maximized = False
        self._build_ui()
        self.refresh_cache_page()
        self.refresh_all()
        self.navigate("recording")

    def _set_initial_window_size(self):
        """Open at a compact size and center the window.

        The size stays unchanged while navigating between tabs. The user can
        still resize or maximize the application normally.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(MIN_APP_WIDTH, MIN_APP_HEIGHT)
            return

        available = screen.availableGeometry()
        width = max(MIN_APP_WIDTH, int(available.width() * WINDOW_SCREEN_RATIO))
        height = max(MIN_APP_HEIGHT, int(available.height() * WINDOW_SCREEN_RATIO))
        width = min(width, MAX_APP_WIDTH, available.width())
        height = min(height, MAX_APP_HEIGHT, available.height())
        self.resize(width, height)

        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav = QFrame()
        self.nav.setFixedWidth(NAV_WIDTH)
        self.nav.setStyleSheet("background:#f8fafc; border-right:1px solid #e5e7eb;")
        nav_layout = QVBoxLayout(self.nav)
        nav_layout.setContentsMargins(14, 16, 14, 16)
        nav_layout.setSpacing(4)

        brand = QLabel("MEA Analyzer")
        brand.setStyleSheet("font-size:22px; font-weight:800; color:#111827; padding:3px 6px 16px 6px;")
        nav_layout.addWidget(brand)
        nav_layout.addWidget(self._section_label("SETUP"))
        for key, label in [
            ("recording", "Recording"),
            ("electrode", "Electrode"),
            ("cache", "Cache"),
        ]:
            self._add_nav_button(nav_layout, key, label)

        nav_layout.addSpacing(16)
        nav_layout.addWidget(self._section_label("ANALYSIS"))
        for i, key in enumerate(ANALYSIS_STEPS, start=1):
            self._add_nav_button(nav_layout, key, f"{i}  {key.title()}")
        nav_layout.addStretch(1)
        self.restart_btn = QPushButton("↻  Restart analysis")
        self.restart_btn.setMinimumHeight(38)
        self.restart_btn.setCursor(Qt.PointingHandCursor)
        self.restart_btn.setStyleSheet(
            "QPushButton {text-align:left; padding:8px 12px; border:1px solid #fca5a5; border-radius:7px; "
            "font-size:12px; color:#b91c1c; background:#fff;}"
            "QPushButton:hover {background:#fef2f2;}"
            "QPushButton:disabled {color:#9ca3af; border-color:#e5e7eb; background:#f9fafb;}"
        )
        self.restart_btn.clicked.connect(self._restart_analysis)
        nav_layout.addWidget(self.restart_btn)
        version = QLabel("v8 • build 20")
        version.setStyleSheet("color:#9ca3af; font-size:11px; padding:6px;")
        nav_layout.addWidget(version)

        outer.addWidget(self.nav)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(8, 14, 8, 14)
        main_layout.setSpacing(8)

        header = QHBoxLayout()
        self.page_heading = QLabel("Recording")
        self.page_heading.setStyleSheet("font-size:24px; font-weight:800; color:#111827;")
        header.addWidget(self.page_heading, 1, Qt.AlignTop)
        self.summary_card = self._build_summary_card()
        header.addWidget(self.summary_card, 0, Qt.AlignRight | Qt.AlignTop)
        main_layout.addLayout(header)

        self.stack = QStackedWidget()
        self.pages["recording"] = self._build_recording_page()
        self.pages["electrode"] = self._build_electrode_page()
        self.pages["cache"] = self._build_cache_page()
        self.pages["range"] = self._build_range_page()
        self.pages["detection"] = self._build_detection_page()
        self.pages["averaging"] = self._build_averaging_page()
        self.pages["baseline"] = self._build_baseline_page()
        self.pages["landmarks"] = self._build_landmarks_page()
        self.pages["results"] = self._build_results_page()
        for step in ANALYSIS_STEPS:
            if step not in self.pages:
                self.pages[step] = self._build_placeholder_page(step)
        for page in self.pages.values():
            self.stack.addWidget(page)
        main_layout.addWidget(self.stack, 1)
        outer.addWidget(main, 1)

    def _section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size:11px; font-weight:800; color:#6b7280; padding:4px 8px;")
        return label

    def _add_nav_button(self, layout, key, label):
        btn = NavButton(label)
        btn.clicked.connect(lambda checked=False, k=key: self.navigate(k))
        self.nav_buttons[key] = btn
        layout.addWidget(btn)

    def _build_summary_card(self):
        # Keep the analysis summary permanently visible, but make it a compact
        # horizontal strip. The recovered vertical space is therefore available
        # to the plots on every analysis page.
        card = QFrame()
        card.setFixedSize(500, 54)
        card.setStyleSheet("QFrame {background:#ffffff; border:1px solid #e5e7eb; border-radius:10px;}")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        title = QLabel("CURRENT\nANALYSIS")
        title.setStyleSheet("font-size:9px; font-weight:800; color:#6b7280; border:none;")
        title.setFixedWidth(54)
        layout.addWidget(title)

        self.summary_file = QLabel("File\n—")
        self.summary_electrode = QLabel("Electrode\n—")
        self.summary_range = QLabel("Range\n—")
        self.summary_threshold = QLabel("Threshold\n—")
        for widget in [self.summary_file, self.summary_electrode, self.summary_range, self.summary_threshold]:
            widget.setStyleSheet("font-size:10px; color:#1f2937; border:none;")
            widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(widget, 1)
        return card

    def _build_recording_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(16)

        info = QLabel(
            "Open the original MEA recording directly. CSV is preferred; XLSX is also supported. "
            "Column 1 is treated as Time and the next up to 64 columns are electrode slots. "
            "Header names can vary. The first opening prepares a fast-access cache."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:14px; color:#374151;")
        layout.addWidget(info)

        file_row = QHBoxLayout()
        self.recording_path = QLineEdit()
        self.recording_path.setReadOnly(True)
        self.recording_path.setPlaceholderText("No recording selected")
        browse = QPushButton("Browse…")
        browse.setMinimumWidth(120)
        browse.clicked.connect(self.choose_recording)
        file_row.addWidget(self.recording_path, 1)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        self.loading_box = QFrame()
        self.loading_box.setStyleSheet("QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px;}")
        box_layout = QVBoxLayout(self.loading_box)
        self.loading_status = QLabel("Select a recording to begin.")
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setValue(0)
        self.loading_progress.setTextVisible(True)
        box_layout.addWidget(self.loading_status)
        box_layout.addWidget(self.loading_progress)
        layout.addWidget(self.loading_box)

        self.recording_details = QLabel("")
        self.recording_details.setStyleSheet("font-size:13px; color:#4b5563;")
        layout.addWidget(self.recording_details)
        layout.addStretch(1)
        return page

    def _build_electrode_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(16)

        grid_host = QWidget()
        self.electrode_grid = QGridLayout(grid_host)
        self.electrode_grid.setContentsMargins(4, 4, 4, 4)
        self.electrode_grid.setHorizontalSpacing(7)
        self.electrode_grid.setVerticalSpacing(7)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll, 3)

        right = QVBoxLayout()
        self.large_preview = LargeSignalPreview()
        right.addWidget(self.large_preview, 1)
        self.use_electrode_btn = QPushButton("Use selected electrode")
        self.use_electrode_btn.setMinimumHeight(44)
        self.use_electrode_btn.setEnabled(False)
        self.use_electrode_btn.clicked.connect(self.confirm_electrode)
        right.addWidget(self.use_electrode_btn)
        layout.addLayout(right, 2)
        return page

    def _build_output_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        text = QLabel(
            "Final export will create one Excel workbook and one PDF report. "
            "Intermediate analysis remains in memory until export."
        )
        text.setWordWrap(True)
        text.setStyleSheet("font-size:14px; color:#374151;")
        layout.addWidget(text)
        layout.addWidget(QLabel("Analysis name"))
        self.analysis_name_edit = QLineEdit()
        self.analysis_name_edit.textChanged.connect(self._output_changed)
        layout.addWidget(self.analysis_name_edit)
        layout.addWidget(QLabel("Save results to"))
        row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(self._output_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_output_dir)
        row.addWidget(self.output_dir_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        self.output_preview = QLabel("")
        self.output_preview.setStyleSheet("font-size:13px; color:#4b5563; margin-top:8px;")
        layout.addWidget(self.output_preview)
        layout.addStretch(1)
        return page

    def _build_cache_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        self.cache_table = QTableWidget(0, 3)
        self.cache_table.setHorizontalHeaderLabels(["Recording", "Last used", "Size"])
        self.cache_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cache_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.cache_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.cache_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cache_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cache_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.cache_table, 1)
        bottom = QHBoxLayout()
        self.cache_total_label = QLabel("Total cache: 0 B")
        bottom.addWidget(self.cache_total_label)
        bottom.addStretch(1)
        clear_selected = QPushButton("Clear selected")
        clear_selected.clicked.connect(self.clear_selected_cache)
        clear_all = QPushButton("Clear cache")
        clear_all.clicked.connect(self.clear_all_cache)
        bottom.addWidget(clear_selected)
        bottom.addWidget(clear_all)
        layout.addLayout(bottom)
        return page

    def _build_range_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        hint = QLabel(
            "Click anywhere to move the nearest boundary, or drag a boundary if you prefer. "
            "Red is always Start/left and blue is End/right. Use the mouse wheel to zoom; double-click to reset the view."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:12px; color:#4b5563;")
        layout.addWidget(hint)

        self.range_plot = RangeSelectionPlot()
        self.range_plot.selectionChanged.connect(self._on_range_plot_changed)
        layout.addWidget(self.range_plot, 1)

        controls = QFrame()
        controls.setStyleSheet(
            "QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}"
        )
        row = QHBoxLayout(controls)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        start_label = QLabel("Start (ms)")
        start_label.setStyleSheet("border:none;")
        self.range_start_spin = QDoubleSpinBox()
        self.range_start_spin.setDecimals(4)
        self.range_start_spin.setRange(-1e12, 1e12)
        self.range_start_spin.setKeyboardTracking(False)
        self.range_start_spin.valueChanged.connect(self._on_range_spin_changed)

        end_label = QLabel("End (ms)")
        end_label.setStyleSheet("border:none;")
        self.range_end_spin = QDoubleSpinBox()
        self.range_end_spin.setDecimals(4)
        self.range_end_spin.setRange(-1e12, 1e12)
        self.range_end_spin.setKeyboardTracking(False)
        self.range_end_spin.valueChanged.connect(self._on_range_spin_changed)

        full_btn = QPushButton("Full recording")
        full_btn.clicked.connect(self._use_full_recording_range)
        view_btn = QPushButton("Reset view")
        view_btn.clicked.connect(self.range_plot.reset_view)

        row.addWidget(start_label)
        row.addWidget(self.range_start_spin)
        row.addSpacing(4)
        row.addWidget(end_label)
        row.addWidget(self.range_end_spin)
        row.addWidget(full_btn)
        row.addWidget(view_btn)
        row.addStretch(1)
        layout.addWidget(controls)

        bottom = QHBoxLayout()
        self.range_info = QLabel("Select an electrode first.")
        self.range_info.setStyleSheet("font-size:12px; color:#4b5563;")
        bottom.addWidget(self.range_info, 1)

        self.range_apply_btn = QPushButton("Use range → Detection")
        self.range_apply_btn.setMinimumHeight(36)
        self.range_apply_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; "
            "padding:7px 16px; font-weight:700;}"
            "QPushButton:hover {background:#1d4ed8;}"
            "QPushButton:disabled {background:#93c5fd; color:#eff6ff;}"
        )
        self.range_apply_btn.clicked.connect(self._confirm_range_and_continue)
        bottom.addWidget(self.range_apply_btn)
        layout.addLayout(bottom)
        return page

    def _prepare_range_page(self):
        if not self.state.selected_electrode:
            return False

        # Range selection is preview-only. Build 20 uses the fast ~5k initial
        # preview immediately, then swaps in a selected-electrode ~30k preview
        # when it becomes available. No full-resolution CSV scan happens here.
        enhanced = load_range_preview(
            self.state.recording, self.state.selected_electrode
        )
        if enhanced is not None:
            time_values = np.asarray(enhanced[0], dtype=np.float64)
            voltage_values = np.asarray(enhanced[1], dtype=np.float64)
        else:
            time_values = np.asarray(self.state.recording.preview_time, dtype=np.float64)
            voltage_values = np.asarray(
                self.state.recording.preview_signals.get(self.state.selected_electrode),
                dtype=np.float64,
            )
        finite_time = time_values[np.isfinite(time_values)]
        if finite_time.size < 2:
            QMessageBox.warning(self, "Range selection", "The recording preview has no usable time axis.")
            return False

        full_start = (
            float(self.state.recording.time_min)
            if self.state.recording.time_min is not None
            else float(np.min(finite_time))
        )
        full_end = (
            float(self.state.recording.time_max)
            if self.state.recording.time_max is not None
            else float(np.max(finite_time))
        )
        if self.state.selected_range is not None:
            start = float(np.clip(self.state.selected_range[0], full_start, full_end))
            end = float(np.clip(self.state.selected_range[1], full_start, full_end))
            if start > end:
                start, end = end, start
        else:
            start, end = full_start, full_end

        for spin in (self.range_start_spin, self.range_end_spin):
            spin.blockSignals(True)
            spin.setRange(full_start, full_end)
        self.range_start_spin.setValue(start)
        self.range_end_spin.setValue(end)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)

        self.range_plot.set_signal(
            time_values, voltage_values, start, end, full_start=full_start, full_end=full_end
        )
        self._update_range_info(start, end, applied=self.state.step_states["range"] == StepState.COMPLETE)
        self._update_range_fast_access_ui()
        if (
            enhanced is None
            and self.state.recording.source_format.casefold() == "csv"
            and self.state.recording.cache_parquet is None
        ):
            QTimer.singleShot(0, lambda e=self.state.selected_electrode: self._start_range_preview_preparation(e))
        return True

    def _on_range_plot_changed(self, start, end):
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setValue(start)
        self.range_end_spin.setValue(end)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)
        self._update_range_info(start, end, applied=False)

    def _on_range_spin_changed(self):
        start = self.range_start_spin.value()
        end = self.range_end_spin.value()
        if start >= end:
            self.range_info.setText("Start must be earlier than End.")
            self.range_info.setStyleSheet("font-size:12px; color:#b91c1c; font-weight:600;")
            return
        self.range_plot.set_selection(start, end, emit=False)
        self._update_range_info(start, end, applied=False)

    def _use_full_recording_range(self):
        preview_time = np.asarray(self.state.recording.preview_time, dtype=np.float64)
        finite = preview_time[np.isfinite(preview_time)]
        if finite.size < 2:
            return
        start = (
            float(self.state.recording.time_min)
            if self.state.recording.time_min is not None
            else float(np.min(finite))
        )
        end = (
            float(self.state.recording.time_max)
            if self.state.recording.time_max is not None
            else float(np.max(finite))
        )
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setValue(start)
        self.range_end_spin.setValue(end)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)
        self.range_plot.set_selection(start, end, emit=False)
        self.range_plot.reset_view()
        self._update_range_info(start, end, applied=False)

    def _update_range_info(self, start, end, applied=False):
        duration = max(0.0, float(end) - float(start))
        status = "Applied" if applied else "Selection not yet applied"
        self.range_info.setStyleSheet(
            "font-size:12px; color:#166534;" if applied else "font-size:12px; color:#4b5563;"
        )
        self.range_info.setText(f"{status}  •  Duration: {duration:,.2f} ms")

    def _confirm_range_and_continue(self):
        """Commit the current Range selection, then advance only after loading succeeds."""
        self._range_continue_after_apply = True
        self._apply_range()

    def _apply_range(self):
        start = float(self.range_start_spin.value())
        end = float(self.range_end_spin.value())
        if start >= end:
            self._range_continue_after_apply = False
            QMessageBox.warning(self, "Invalid range", "Start must be earlier than End.")
            return
        if not self.state.selected_electrode:
            self._range_continue_after_apply = False
            QMessageBox.information(self, "Select an electrode", "Choose an electrode first.")
            return
        if self._range_load_thread is not None:
            self._range_continue_after_apply = False
            return

        electrode = self.state.selected_electrode
        recording = self.state.recording
        lazy_csv = (
            recording.source_format.casefold() == "csv"
            and recording.cache_parquet is None
            and not fast_electrode_cache_ready(recording, electrode)
        )

        if lazy_csv:
            # Build 20: only now do we touch full-resolution CSV data, and only
            # for the selected interval. This runs off the GUI thread so the app
            # remains responsive while the exact samples are collected.
            self._start_range_load_worker(electrode, start, end)
            return

        # Existing XLSX / completed-sidecar path. Read only the selected interval.
        self.range_info.setStyleSheet("font-size:12px; color:#4b5563;")
        self.range_info.setText("Loading only the selected interval at full resolution…")
        self.range_apply_btn.setEnabled(False)
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            time_values, voltage_values = load_electrode_range(
                recording, electrode, start, end
            )
        except Exception as exc:
            self._range_continue_after_apply = False
            QMessageBox.critical(self, "Range loading error", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._update_range_fast_access_ui()

        self._commit_loaded_range(time_values, voltage_values, start, end, electrode)

    def _start_range_load_worker(self, electrode: str, start: float, end: float):
        if self._range_load_thread is not None:
            return
        self._range_load_progress = 0
        self.range_info.setStyleSheet("font-size:12px; color:#4b5563;")
        self.range_info.setText("Loading selected interval at full resolution…")

        thread = QThread(self)
        worker = RangeLoadWorker(self.state.recording, electrode, start, end)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_range_load_progress)
        worker.finished.connect(self._on_range_load_ready)
        worker.failed.connect(self._on_range_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._range_load_thread_finished)
        self._range_load_thread = thread
        self._range_load_worker = worker
        thread.start()
        self._update_range_fast_access_ui()

    def _on_range_load_progress(self, pct: int):
        self._range_load_progress = max(0, min(99, int(pct)))
        self._update_range_fast_access_ui()

    def _on_range_load_ready(self, time_values, voltage_values, start: float, end: float, electrode: str):
        # If the user changed electrode/range while the worker was running, do not
        # silently apply stale data. They can simply click Apply again.
        current_start = float(self.range_start_spin.value())
        current_end = float(self.range_end_spin.value())
        if (
            electrode != self.state.selected_electrode
            or not np.isclose(current_start, float(start), rtol=0.0, atol=1e-9)
            or not np.isclose(current_end, float(end), rtol=0.0, atol=1e-9)
        ):
            self.range_info.setStyleSheet("font-size:12px; color:#4b5563;")
            self._range_continue_after_apply = False
            self.range_info.setText("Range selection changed while loading. Confirm the range again.")
            return
        self._commit_loaded_range(time_values, voltage_values, start, end, electrode)

    def _on_range_load_failed(self, details: str):
        self._range_continue_after_apply = False
        QMessageBox.critical(self, "Range loading error", details)
        self.range_info.setStyleSheet("font-size:12px; color:#b91c1c; font-weight:600;")
        self.range_info.setText("Could not load the selected interval at full resolution.")

    def _range_load_thread_finished(self):
        self._range_load_thread = None
        self._range_load_worker = None
        self._range_load_progress = 0
        self._update_range_fast_access_ui()

    def _commit_loaded_range(self, time_values, voltage_values, start: float, end: float, electrode: str):
        if electrode != self.state.selected_electrode:
            return
        count = int(len(time_values))
        if count < 2:
            QMessageBox.warning(self, "Invalid range", "The selected range contains fewer than two samples.")
            return

        # Same v7 Cell 2.5 scientific selection: start <= time <= end. The full
        # arrays remain full resolution for smoothing/Z-score/beat detection.
        # Plot widgets independently decimate their display copy for responsiveness.
        self.state.selected_range = (float(start), float(end))
        self.state.range_time = np.asarray(time_values, dtype=np.float64)
        self.state.range_voltage_raw = np.asarray(voltage_values, dtype=np.float64)
        self.state.invalidate_from("range")
        self._update_range_info(start, end, applied=True)
        self.range_info.setText(
            f"Applied  •  {count:,} full-resolution points loaded  •  Duration: {end - start:,.2f} ms"
        )
        self.refresh_all()
        if self._range_continue_after_apply:
            self._range_continue_after_apply = False
            QTimer.singleShot(0, lambda: self.navigate("detection"))

    def _build_detection_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        hint = QLabel(
            "Detection uses only the full-resolution data from the applied range. "
            "Smoothing reproduces v7 Cell 2.75 (0 ms = raw). Click or drag in the Z-score panel to set the threshold."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:12px; color:#4b5563;")
        layout.addWidget(hint)

        self.detection_plot = DetectionPlot()
        self.detection_plot.thresholdChanged.connect(self._on_detection_plot_threshold)
        layout.addWidget(self.detection_plot, 1)

        controls = QFrame()
        controls.setStyleSheet(
            "QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}"
        )
        row = QHBoxLayout(controls)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        smooth_label = QLabel("Smoothing (ms)")
        smooth_label.setStyleSheet("border:none;")
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setDecimals(2)
        self.smoothing_spin.setRange(0.0, 1000.0)
        self.smoothing_spin.setSingleStep(1.0)
        self.smoothing_spin.setKeyboardTracking(False)
        self.smoothing_spin.setValue(10.0)
        self.smoothing_spin.valueChanged.connect(self._on_detection_smoothing_changed)

        threshold_label = QLabel("Z threshold")
        threshold_label.setStyleSheet("border:none;")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setRange(-100.0, 100.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setKeyboardTracking(False)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.valueChanged.connect(self._on_detection_threshold_spin)

        row.addWidget(smooth_label)
        row.addWidget(self.smoothing_spin)
        row.addSpacing(8)
        row.addWidget(threshold_label)
        row.addWidget(self.threshold_spin)
        row.addStretch(1)
        layout.addWidget(controls)

        bottom = QHBoxLayout()
        self.detection_info = QLabel("Confirm a range first.")
        self.detection_info.setStyleSheet("font-size:12px; color:#4b5563;")
        bottom.addWidget(self.detection_info, 1)

        back_range_btn = QPushButton("← Range")
        back_range_btn.clicked.connect(lambda: self.navigate("range"))
        bottom.addWidget(back_range_btn)

        apply_btn = QPushButton("Use detection → Averaging")
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; "
            "padding:7px 16px; font-weight:700;}"
            "QPushButton:hover {background:#1d4ed8;}"
        )
        apply_btn.clicked.connect(self._confirm_detection_and_continue)
        bottom.addWidget(apply_btn)
        layout.addLayout(bottom)
        return page

    def _prepare_detection_page(self):
        if self.state.range_time is None or self.state.range_voltage_raw is None:
            QMessageBox.information(
                self,
                "Apply a range first",
                "Choose the analysis interval on the Range page and confirm it before continuing.",
            )
            return False

        self.smoothing_spin.blockSignals(True)
        self.threshold_spin.blockSignals(True)
        self.smoothing_spin.setValue(float(self.state.filter_window_ms))
        self.threshold_spin.setValue(3.0 if self.state.threshold is None else float(self.state.threshold))
        self.smoothing_spin.blockSignals(False)
        self.threshold_spin.blockSignals(False)
        return self._recalculate_detection_preview(mark_dirty=False)

    def _recalculate_detection_preview(self, mark_dirty=True):
        if self.state.range_time is None or self.state.range_voltage_raw is None:
            return False
        filter_window_ms = float(self.smoothing_spin.value())
        threshold = float(self.threshold_spin.value())
        try:
            filtered, dt_ms, window_samples = apply_v7_smoothing(
                self.state.range_time,
                self.state.range_voltage_raw,
                filter_window_ms,
            )
            z_scores = calculate_z_score_v7(filtered)
            spike_times, spike_indices = detect_spikes_v7(
                z_scores, self.state.range_time, threshold
            )
        except Exception as exc:
            self.detection_info.setStyleSheet("font-size:12px; color:#b91c1c; font-weight:600;")
            self.detection_info.setText(str(exc))
            return False

        self.state.filter_window_ms = filter_window_ms
        self.state.threshold = threshold
        self.state.detection_voltage_filtered = filtered
        self.state.detection_z_scores = z_scores
        self.state.spike_timepoints = spike_times
        self.state.spike_indices = spike_indices
        self.state.detection_window_samples = int(window_samples)
        self.state.detection_dt_ms = float(dt_ms)

        self.detection_plot.set_data(
            self.state.range_time,
            self.state.range_voltage_raw,
            filtered,
            z_scores,
            threshold,
            spike_indices,
            smoothing_enabled=filter_window_ms > 0,
        )
        self._update_detection_info(applied=self.state.step_states["detection"] == StepState.COMPLETE)
        if mark_dirty:
            self.state.mark_step_dirty("detection")
            self.refresh_all()
        return True

    def _on_detection_smoothing_changed(self):
        self._recalculate_detection_preview(mark_dirty=True)

    def _on_detection_threshold_spin(self):
        self._update_detection_threshold(float(self.threshold_spin.value()), mark_dirty=True)

    def _on_detection_plot_threshold(self, value):
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(float(value))
        self.threshold_spin.blockSignals(False)
        self._update_detection_threshold(float(value), mark_dirty=True)

    def _update_detection_threshold(self, threshold, mark_dirty=True):
        if self.state.detection_z_scores is None or self.state.range_time is None:
            self._recalculate_detection_preview(mark_dirty=mark_dirty)
            return
        spike_times, spike_indices = detect_spikes_v7(
            self.state.detection_z_scores,
            self.state.range_time,
            float(threshold),
        )
        self.state.threshold = float(threshold)
        self.state.spike_timepoints = spike_times
        self.state.spike_indices = spike_indices
        self.detection_plot.set_threshold(float(threshold), redraw=False)
        self.detection_plot.set_spikes(spike_indices, redraw=True)
        if mark_dirty:
            self.state.mark_step_dirty("detection")
        self._update_detection_info(applied=False)
        self.refresh_all()

    def _update_detection_info(self, applied=False):
        count = 0 if self.state.spike_indices is None else int(len(self.state.spike_indices))
        dt = self.state.detection_dt_ms
        if self.state.filter_window_ms <= 0:
            smooth_text = "raw signal (no smoothing)"
        else:
            smooth_text = (
                f"SG {self.state.detection_window_samples} samples"
                + (f" ≈ {self.state.detection_window_samples * dt:.2f} ms" if dt else "")
            )
        status = "Applied" if applied else "Preview"
        self.detection_info.setStyleSheet(
            "font-size:12px; color:#166534;" if applied else "font-size:12px; color:#4b5563;"
        )
        self.detection_info.setText(f"{status}  •  {count:,} threshold crossings  •  {smooth_text}")

    def _apply_detection(self):
        if not self._recalculate_detection_preview(mark_dirty=False):
            return
        count = 0 if self.state.spike_indices is None else int(len(self.state.spike_indices))
        if count == 0:
            QMessageBox.warning(
                self,
                "No spikes detected",
                "No positive Z-score threshold crossings were detected. Adjust the threshold or smoothing before continuing.",
            )
            self.state.mark_step_dirty("detection")
            self.refresh_all()
            return
        self.state.invalidate_from("detection")
        self._update_detection_info(applied=True)
        self.refresh_all()

    def _confirm_detection_and_continue(self):
        self._apply_detection()
        if self.state.step_states["detection"] == StepState.COMPLETE:
            self.navigate("averaging")

    def _build_averaging_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        hint = QLabel(
            "Each group contains consecutive detected beats. The selected averaging factor is the number of beats "
            "combined into one mean waveform. The scientific calculation is unchanged from v7; the word ‘group’ "
            "is used here instead of ‘block’ for clarity. Use the colored beat checkboxes on the right (or click a trace) to exclude/include it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:12px; color:#4b5563;")
        layout.addWidget(hint)

        controls = QFrame()
        controls.setStyleSheet("QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}")
        row = QHBoxLayout(controls)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)
        factor_label = QLabel("Beats per average")
        factor_label.setStyleSheet("border:none;")
        self.averaging_factor_spin = QSpinBox()
        self.averaging_factor_spin.setRange(1, 1)
        self.averaging_factor_spin.setSingleStep(1)
        self.averaging_factor_spin.setKeyboardTracking(False)
        self.averaging_factor_spin.setWrapping(False)
        self.averaging_factor_spin.valueChanged.connect(self._on_averaging_factor_changed)
        self.averaging_detected_label = QLabel("Detected peaks: —")
        self.averaging_detected_label.setStyleSheet("border:none; color:#4b5563;")
        self.update_averaging_btn = QPushButton("Update averaging")
        self.update_averaging_btn.setToolTip("Rebuild the beat groups and mean waveforms using the selected averaging factor.")
        self.update_averaging_btn.clicked.connect(self._prepare_averaging_blocks)
        row.addWidget(factor_label)
        row.addWidget(self.averaging_factor_spin)
        row.addSpacing(8)
        row.addWidget(self.averaging_detected_label)
        row.addStretch(1)
        row.addWidget(self.update_averaging_btn)
        layout.addWidget(controls)

        self.averaging_plot = SnippetAveragingPlot()
        self.averaging_plot.selectionChanged.connect(self._on_averaging_snippet_toggled)
        layout.addWidget(self.averaging_plot, 1)

        block_row = QHBoxLayout()
        self.prev_block_btn = QPushButton("← Previous group")
        self.prev_block_btn.clicked.connect(lambda: self._change_averaging_block(-1))
        self.next_block_btn = QPushButton("Next group →")
        self.next_block_btn.clicked.connect(lambda: self._change_averaging_block(1))
        self.averaging_block_label = QLabel("No groups prepared")
        self.averaging_block_label.setAlignment(Qt.AlignCenter)
        self.averaging_block_label.setStyleSheet("font-size:12px; font-weight:700; color:#374151;")
        block_row.addWidget(self.prev_block_btn)
        block_row.addWidget(self.averaging_block_label, 1)
        block_row.addWidget(self.next_block_btn)
        layout.addLayout(block_row)

        bottom = QHBoxLayout()
        self.averaging_info = QLabel("Confirm Detection first.")
        self.averaging_info.setStyleSheet("font-size:12px; color:#4b5563;")
        bottom.addWidget(self.averaging_info, 1)

        back_detection_btn = QPushButton("← Detection")
        back_detection_btn.clicked.connect(lambda: self.navigate("detection"))
        bottom.addWidget(back_detection_btn)

        reset_btn = QPushButton("Restore all beats")
        reset_btn.setToolTip("Re-include every beat in the current group after you excluded one or more traces.")
        reset_btn.clicked.connect(self._reset_current_block_selection)
        bottom.addWidget(reset_btn)

        apply_btn = QPushButton("Use averaging → Baseline")
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; "
            "padding:7px 16px; font-weight:700;}"
            "QPushButton:hover {background:#1d4ed8;}"
        )
        apply_btn.clicked.connect(self._confirm_averaging_and_continue)
        bottom.addWidget(apply_btn)
        layout.addLayout(bottom)

        self._averaging_current_block = 1
        self._averaging_factor_pending = False
        return page

    def _prepare_averaging_page(self):
        if self.state.step_states["detection"] != StepState.COMPLETE:
            QMessageBox.information(
                self,
                "Apply Detection first",
                "Confirm the current smoothing and Z-score threshold on the Detection page before averaging.",
            )
            return False
        count = 0 if self.state.spike_timepoints is None else int(len(self.state.spike_timepoints))
        if count == 0:
            QMessageBox.warning(self, "No detected peaks", "No detected peaks are available for averaging.")
            return False

        self.averaging_factor_spin.blockSignals(True)
        self.averaging_factor_spin.setRange(1, count)
        if self.state.averaging_factor is None:
            # UI default only. v7 required manual selection; using 5 gives an immediately useful
            # preview while preserving the exact v7 grouping/averaging calculation.
            default_factor = min(5, count)
            self.averaging_factor_spin.setValue(default_factor)
        else:
            self.averaging_factor_spin.setValue(min(max(1, int(self.state.averaging_factor)), count))
        self.averaging_factor_spin.blockSignals(False)
        self.averaging_detected_label.setText(f"Detected peaks: {count:,}  •  Allowed factor: 1–{count:,}")

        if self.state.averaging_factor is None and not self.state.averaging_blocks:
            # Show a useful result immediately instead of making the user press a separate
            # 'prepare blocks' button on first entry.
            return self._prepare_averaging_blocks(mark_dirty=False)

        needs_rebuild = self.state.step_states["averaging"] == StepState.NEEDS_UPDATE
        if needs_rebuild:
            return self._prepare_averaging_blocks(mark_dirty=False)

        self._averaging_current_block = min(
            max(1, getattr(self, "_averaging_current_block", 1)),
            max(1, len(self.state.averaging_blocks)),
        )
        self._show_averaging_block(self._averaging_current_block)
        self._update_averaging_info(applied=self.state.step_states["averaging"] == StepState.COMPLETE)
        return True

    def _on_averaging_factor_changed(self):
        self._averaging_factor_pending = True
        if self.state.step_states["averaging"] in {StepState.COMPLETE, StepState.NEEDS_UPDATE}:
            self.state.mark_step_dirty("averaging")
            self.refresh_all()
        self.averaging_info.setStyleSheet("font-size:12px; color:#92400e;")
        self.averaging_info.setText("Averaging factor changed — click Update averaging to rebuild the groups and mean waveforms.")

    def _prepare_averaging_blocks(self, mark_dirty=True):
        if self.state.spike_timepoints is None or self.state.detection_voltage_filtered is None:
            return False
        factor = int(self.averaging_factor_spin.value())
        try:
            blocks, discarded = divide_spikes_into_blocks_v7(self.state.spike_timepoints, factor)
            if not blocks:
                raise ValueError("This averaging factor does not create any complete block.")
            isi_results = compute_isi_results_v7(blocks)

            snippets_by_block = {}
            times_by_block = {}
            valid_by_block = {}
            masks_by_block = {}
            means_by_block = {}
            lengths_by_block = {}
            for block_idx, spike_times in blocks.items():
                snippets, snippet_times, valid_spikes, snippet_length = extract_snippets_v7(
                    self.state.range_time,
                    self.state.detection_voltage_filtered,
                    spike_times,
                    x_ms=20.0,
                    max_extension=1700.0,
                )
                if not snippets:
                    continue
                mask = np.ones(len(snippets), dtype=bool)
                mean = compute_selected_mean_v7(snippets, mask)
                snippets_by_block[int(block_idx)] = snippets
                times_by_block[int(block_idx)] = snippet_times
                valid_by_block[int(block_idx)] = valid_spikes
                masks_by_block[int(block_idx)] = mask
                means_by_block[int(block_idx)] = mean
                lengths_by_block[int(block_idx)] = float(snippet_length)

            if not snippets_by_block:
                raise ValueError("No valid snippets could be extracted from the prepared blocks.")
        except Exception as exc:
            QMessageBox.warning(self, "Averaging", str(exc))
            return False

        self.state.averaging_factor = factor
        self.state.averaging_blocks = {int(k): list(v) for k, v in blocks.items()}
        self.state.averaging_isi_results = isi_results
        self.state.averaging_snippets = snippets_by_block
        self.state.averaging_snippet_times = times_by_block
        self.state.averaging_valid_spikes = valid_by_block
        self.state.averaging_selected_masks = masks_by_block
        self.state.averaging_mean_snippets = means_by_block
        self.state.averaging_snippet_lengths_ms = lengths_by_block
        if self.state.baseline_values and self.state.step_states["baseline"] == StepState.NOT_STARTED:
            self.state.step_states["baseline"] = StepState.NEEDS_UPDATE
        self._averaging_current_block = min(snippets_by_block.keys())
        self._averaging_factor_pending = False

        if mark_dirty:
            self.state.mark_step_dirty("averaging")
        self._show_averaging_block(self._averaging_current_block)
        self._update_averaging_info(applied=False, discarded=discarded)
        self.refresh_all()
        return True

    def _show_averaging_block(self, block_idx):
        if not self.state.averaging_snippets:
            self.averaging_block_label.setText("No groups prepared")
            return
        keys = sorted(self.state.averaging_snippets)
        if block_idx not in keys:
            block_idx = keys[0]
        self._averaging_current_block = int(block_idx)
        snippets = self.state.averaging_snippets[block_idx]
        times = self.state.averaging_snippet_times[block_idx]
        mask = self.state.averaging_selected_masks[block_idx]
        mean = self.state.averaging_mean_snippets[block_idx]
        length_ms = self.state.averaging_snippet_lengths_ms[block_idx]
        self.averaging_plot.set_block(block_idx, times, snippets, mask, mean, length_ms)

        included = int(np.sum(mask))
        total = len(mask)
        isi_mean, isi_std = self.state.averaging_isi_results.get(block_idx, (None, None))
        if isi_mean is None:
            isi_text = "ISI: n/a"
        else:
            isi_text = f"ISI: {isi_mean:.2f} ± {isi_std:.2f} ms"
        position = keys.index(block_idx) + 1
        self.averaging_block_label.setText(
            f"Group {block_idx} ({position}/{len(keys)})  •  Included {included}/{total} beats  •  {isi_text}"
        )
        self.prev_block_btn.setEnabled(position > 1)
        self.next_block_btn.setEnabled(position < len(keys))

    def _change_averaging_block(self, direction):
        keys = sorted(self.state.averaging_snippets)
        if not keys:
            return
        try:
            pos = keys.index(self._averaging_current_block)
        except ValueError:
            pos = 0
        pos = max(0, min(len(keys) - 1, pos + int(direction)))
        self._show_averaging_block(keys[pos])

    def _on_averaging_snippet_toggled(self, snippet_idx, selected):
        block_idx = getattr(self, "_averaging_current_block", None)
        if block_idx not in self.state.averaging_selected_masks:
            return
        mask = self.state.averaging_selected_masks[block_idx]
        if snippet_idx < 0 or snippet_idx >= len(mask):
            return
        mask[snippet_idx] = bool(selected)
        snippets = self.state.averaging_snippets[block_idx]
        mean = compute_selected_mean_v7(snippets, mask)
        self.state.averaging_mean_snippets[block_idx] = mean
        self.averaging_plot.update_mean(mask, mean)
        self._show_averaging_block(block_idx)
        self.state.mark_step_dirty("averaging")
        if self.state.baseline_values and self.state.step_states["baseline"] == StepState.NOT_STARTED:
            self.state.step_states["baseline"] = StepState.NEEDS_UPDATE
        self._update_averaging_info(applied=False)
        self.refresh_all()

    def _reset_current_block_selection(self):
        block_idx = getattr(self, "_averaging_current_block", None)
        if block_idx not in self.state.averaging_selected_masks:
            return
        mask = np.ones(len(self.state.averaging_selected_masks[block_idx]), dtype=bool)
        self.state.averaging_selected_masks[block_idx] = mask
        mean = compute_selected_mean_v7(self.state.averaging_snippets[block_idx], mask)
        self.state.averaging_mean_snippets[block_idx] = mean
        self._show_averaging_block(block_idx)
        self.state.mark_step_dirty("averaging")
        if self.state.baseline_values and self.state.step_states["baseline"] == StepState.NOT_STARTED:
            self.state.step_states["baseline"] = StepState.NEEDS_UPDATE
        self._update_averaging_info(applied=False)
        self.refresh_all()

    def _update_averaging_info(self, applied=False, discarded=None):
        blocks = len(self.state.averaging_snippets)
        factor = self.state.averaging_factor
        status = "Applied" if applied else "Prepared"
        suffix = ""
        if discarded:
            suffix = f"  •  {discarded} final beat(s) not used (incomplete group)"
        self.averaging_info.setStyleSheet(
            "font-size:12px; color:#166534;" if applied else "font-size:12px; color:#4b5563;"
        )
        self.averaging_info.setText(
            f"{status}  •  {factor if factor is not None else '—'} beats/group  •  {blocks} complete group(s){suffix}"
        )

    def _apply_averaging(self):
        if self._averaging_factor_pending:
            if not self._prepare_averaging_blocks(mark_dirty=False):
                return
        if not self.state.averaging_snippets:
            QMessageBox.information(
                self,
                "No averaging groups",
                "No averaging groups are available. Review Detection and update averaging first.",
            )
            return

        empty_blocks = [
            idx for idx, mask in self.state.averaging_selected_masks.items() if int(np.sum(mask)) == 0
        ]
        if empty_blocks:
            QMessageBox.warning(
                self,
                "No snippets selected",
                "At least one snippet must remain included in every block. "
                f"Check block(s): {', '.join(map(str, empty_blocks))}.",
            )
            return

        self.state.invalidate_from("averaging")
        self._update_averaging_info(applied=True)
        self._show_averaging_block(self._averaging_current_block)
        self.refresh_all()

    def _confirm_averaging_and_continue(self):
        self._apply_averaging()
        if self.state.step_states["averaging"] == StepState.COMPLETE:
            self.navigate("baseline")

    def _build_baseline_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        hint = QLabel(
            "Baseline is calculated separately for each averaged group using the frozen v7 method. "
            "Blue = pre-spike baseline, red = post-spike baseline, black = final baseline. "
            "As in v7, the final baseline starts at the pre-spike value; click anywhere on the plot to move it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:12px; color:#4b5563;")
        layout.addWidget(hint)

        self.baseline_plot = BaselinePlot()
        self.baseline_plot.baselineChanged.connect(self._on_baseline_plot_changed)
        layout.addWidget(self.baseline_plot, 1)

        group_row = QHBoxLayout()
        self.prev_baseline_btn = QPushButton("← Previous group")
        self.prev_baseline_btn.clicked.connect(lambda: self._change_baseline_group(-1))
        self.next_baseline_btn = QPushButton("Next group →")
        self.next_baseline_btn.clicked.connect(lambda: self._change_baseline_group(1))
        self.baseline_group_label = QLabel("No groups available")
        self.baseline_group_label.setAlignment(Qt.AlignCenter)
        self.baseline_group_label.setStyleSheet("font-size:12px; font-weight:700; color:#374151;")
        group_row.addWidget(self.prev_baseline_btn)
        group_row.addWidget(self.baseline_group_label, 1)
        group_row.addWidget(self.next_baseline_btn)
        layout.addLayout(group_row)

        details = QFrame()
        details.setStyleSheet("QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}")
        details_row = QHBoxLayout(details)
        details_row.setContentsMargins(10, 7, 10, 7)
        details_row.setSpacing(8)
        self.baseline_details = QLabel("Baseline details: —")
        self.baseline_details.setStyleSheet("border:none; color:#4b5563; font-size:11px;")
        self.baseline_details.setWordWrap(True)
        details_row.addWidget(self.baseline_details, 1)

        exact_label = QLabel("Final baseline")
        exact_label.setStyleSheet("border:none;")
        self.baseline_value_spin = QDoubleSpinBox()
        self.baseline_value_spin.setDecimals(6)
        self.baseline_value_spin.setRange(-1e6, 1e6)
        self.baseline_value_spin.setSingleStep(0.001)
        self.baseline_value_spin.setKeyboardTracking(False)
        self.baseline_value_spin.valueChanged.connect(self._on_baseline_spin_changed)
        details_row.addWidget(exact_label)
        details_row.addWidget(self.baseline_value_spin)
        layout.addWidget(details)

        bottom = QHBoxLayout()
        self.baseline_info = QLabel("Confirm Averaging first.")
        self.baseline_info.setStyleSheet("font-size:12px; color:#4b5563;")
        bottom.addWidget(self.baseline_info, 1)

        back_btn = QPushButton("← Averaging")
        back_btn.clicked.connect(lambda: self.navigate("averaging"))
        bottom.addWidget(back_btn)

        pre_btn = QPushButton("Use pre")
        pre_btn.setToolTip("Set the final baseline to the v7 pre-spike baseline for this group.")
        pre_btn.clicked.connect(lambda: self._set_current_baseline_reference("pre"))
        bottom.addWidget(pre_btn)

        post_btn = QPushButton("Use post")
        post_btn.setToolTip("Set the final baseline to the v7 post-spike baseline for this group.")
        post_btn.clicked.connect(lambda: self._set_current_baseline_reference("post"))
        bottom.addWidget(post_btn)

        apply_btn = QPushButton("Use baseline → Landmarks")
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; "
            "padding:7px 16px; font-weight:700;}"
            "QPushButton:hover {background:#1d4ed8;}"
        )
        apply_btn.clicked.connect(self._confirm_baseline_and_continue)
        bottom.addWidget(apply_btn)
        layout.addLayout(bottom)

        self._baseline_current_group = 1
        return page

    def _prepare_baseline_page(self):
        if self.state.step_states["averaging"] != StepState.COMPLETE:
            QMessageBox.information(
                self,
                "Apply Averaging first",
                "Review the averaging groups and confirm them before baseline selection.",
            )
            return False
        if not self.state.averaging_mean_snippets:
            QMessageBox.warning(self, "No mean waveforms", "No averaged mean waveforms are available for baseline selection.")
            return False

        expected_keys = set(self.state.averaging_mean_snippets.keys())
        stored_keys = set(self.state.baseline_values.keys())
        needs_rebuild = self.state.step_states["baseline"] == StepState.NEEDS_UPDATE
        if needs_rebuild or stored_keys != expected_keys or not stored_keys:
            if not self._calculate_baselines_from_averages():
                return False

        keys = sorted(self.state.baseline_values)
        if not keys:
            return False
        current = getattr(self, "_baseline_current_group", keys[0])
        self._baseline_current_group = current if current in keys else keys[0]
        self._show_baseline_group(self._baseline_current_group)
        self._update_baseline_info(applied=self.state.step_states["baseline"] == StepState.COMPLETE)
        return True

    def _calculate_baselines_from_averages(self):
        baseline_values = {}
        baseline_pre = {}
        baseline_post = {}
        baseline_best = {}
        selected_regions = {}
        region_pre = {}
        region_post = {}
        std_pre = {}
        std_post = {}

        try:
            for group_idx in sorted(self.state.averaging_mean_snippets):
                time = self.state.averaging_snippet_times[group_idx]
                mean = self.state.averaging_mean_snippets[group_idx]
                (
                    best,
                    pre,
                    post,
                    pre_region,
                    post_region,
                    selected_region,
                    pre_std,
                    post_std,
                ) = determine_baseline_v7(time, mean, pre_spike_fraction=0.5, post_spike_window=50.0)

                if not np.isfinite(pre) or not np.isfinite(post):
                    raise ValueError(f"Group {group_idx} produced a non-finite v7 baseline.")
                baseline_pre[int(group_idx)] = float(pre)
                baseline_post[int(group_idx)] = float(post)
                baseline_best[int(group_idx)] = float(best)
                selected_regions[int(group_idx)] = str(selected_region)
                region_pre[int(group_idx)] = tuple(pre_region)
                region_post[int(group_idx)] = tuple(post_region)
                std_pre[int(group_idx)] = float(pre_std)
                std_post[int(group_idx)] = float(post_std)

                # Important v7 detail: InteractiveBaselinePlot initializes the
                # final baseline at baseline_pre, even though determine_baseline
                # also reports which region has the lower standard deviation.
                baseline_values[int(group_idx)] = float(pre)
        except Exception as exc:
            QMessageBox.warning(self, "Baseline", str(exc))
            return False

        self.state.baseline_values = baseline_values
        self.state.baseline_pre = baseline_pre
        self.state.baseline_post = baseline_post
        self.state.baseline_best = baseline_best
        self.state.baseline_selected_region = selected_regions
        self.state.baseline_region_pre = region_pre
        self.state.baseline_region_post = region_post
        self.state.baseline_std_pre = std_pre
        self.state.baseline_std_post = std_post
        self._baseline_current_group = min(baseline_values)
        return True

    def _show_baseline_group(self, group_idx):
        keys = sorted(self.state.baseline_values)
        if not keys:
            self.baseline_group_label.setText("No groups available")
            return
        if group_idx not in keys:
            group_idx = keys[0]
        self._baseline_current_group = int(group_idx)

        time = self.state.averaging_snippet_times[group_idx]
        mean = self.state.averaging_mean_snippets[group_idx]
        pre = self.state.baseline_pre[group_idx]
        post = self.state.baseline_post[group_idx]
        final = self.state.baseline_values[group_idx]
        self.baseline_plot.set_group(group_idx, time, mean, pre, post, final)

        self.baseline_value_spin.blockSignals(True)
        self.baseline_value_spin.setValue(float(final))
        self.baseline_value_spin.blockSignals(False)

        position = keys.index(group_idx) + 1
        self.baseline_group_label.setText(f"Group {group_idx} ({position}/{len(keys)})")
        self.prev_baseline_btn.setEnabled(position > 1)
        self.next_baseline_btn.setEnabled(position < len(keys))

        stable = self.state.baseline_selected_region.get(group_idx, "—")
        pre_std = self.state.baseline_std_pre.get(group_idx, float("nan"))
        post_std = self.state.baseline_std_post.get(group_idx, float("nan"))
        self.baseline_details.setText(
            f"Pre: {pre:.6g} mV (σ {pre_std:.4g})   •   "
            f"Post: {post:.6g} mV (σ {post_std:.4g})   •   More stable in v7 calculation: {stable}"
        )

    def _change_baseline_group(self, direction):
        keys = sorted(self.state.baseline_values)
        if not keys:
            return
        try:
            pos = keys.index(self._baseline_current_group)
        except ValueError:
            pos = 0
        pos = max(0, min(len(keys) - 1, pos + int(direction)))
        self._show_baseline_group(keys[pos])

    def _set_baseline_value(self, value):
        group_idx = getattr(self, "_baseline_current_group", None)
        if group_idx not in self.state.baseline_values or not np.isfinite(value):
            return
        self.state.baseline_values[group_idx] = float(value)
        self.baseline_plot.set_baseline(float(value), emit=False)
        self.baseline_value_spin.blockSignals(True)
        self.baseline_value_spin.setValue(float(value))
        self.baseline_value_spin.blockSignals(False)
        self.state.mark_step_dirty("baseline")
        self._update_baseline_info(applied=False)
        self.refresh_all()

    def _on_baseline_plot_changed(self, value):
        self._set_baseline_value(float(value))

    def _on_baseline_spin_changed(self, value):
        self._set_baseline_value(float(value))

    def _set_current_baseline_reference(self, which):
        group_idx = getattr(self, "_baseline_current_group", None)
        if group_idx not in self.state.baseline_values:
            return
        if which == "post":
            value = self.state.baseline_post[group_idx]
        else:
            value = self.state.baseline_pre[group_idx]
        self._set_baseline_value(float(value))

    def _update_baseline_info(self, applied=False):
        count = len(self.state.baseline_values)
        status = "Applied" if applied else "Ready"
        self.baseline_info.setStyleSheet(
            "font-size:12px; color:#166534;" if applied else "font-size:12px; color:#4b5563;"
        )
        self.baseline_info.setText(f"{status}  •  baselines available for {count} group(s)")

    def _apply_baseline(self):
        expected = set(self.state.averaging_mean_snippets.keys())
        available = set(self.state.baseline_values.keys())
        if not expected or available != expected:
            QMessageBox.warning(self, "Baseline", "Baseline values are not available for every averaging group.")
            return
        bad = [idx for idx, value in self.state.baseline_values.items() if not np.isfinite(value)]
        if bad:
            QMessageBox.warning(
                self,
                "Baseline",
                f"Invalid baseline value in group(s): {', '.join(map(str, bad))}.",
            )
            return
        # Landmark positions depend on the chosen baseline. Clear only that downstream
        # derived state here; manual landmark edits are otherwise preserved when revisiting.
        self.state.landmark_values = {}
        self.state.landmark_auto_values = {}
        self.state.landmark_noise_floor = {}
        self.state.landmark_flags = {}
        self.state.landmark_enabled = {}
        self.state.invalidate_from("baseline")
        self._update_baseline_info(applied=True)
        self._show_baseline_group(self._baseline_current_group)
        self.refresh_all()

    def _confirm_baseline_and_continue(self):
        self._apply_baseline()
        if self.state.step_states["baseline"] == StepState.COMPLETE:
            self.navigate("landmarks")

    def _build_landmarks_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(7)

        hint = QLabel(
            "Automatic marker positions still use the frozen v7 calculation. The checkbox controls whether a marker is "
            "visible/included; its colored name selects the marker you are adjusting. Hidden markers keep their calculated "
            "value. Full view is the default. Use the zoom control to focus on the active marker, then drag the scrollbar below the graph to move across the waveform. Mouse wheel zooms further."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:11px; color:#4b5563;")
        layout.addWidget(hint)

        content = QHBoxLayout()
        content.setSpacing(4)

        plot_column = QVBoxLayout()
        plot_column.setContentsMargins(0, 0, 0, 0)
        plot_column.setSpacing(3)
        plot_tools = QHBoxLayout()
        plot_tools.setContentsMargins(0, 0, 0, 0)
        plot_tools.addStretch(1)
        self.landmark_zoom_icon_btn = QPushButton("🔍")
        self.landmark_zoom_icon_btn.setCheckable(True)
        self.landmark_zoom_icon_btn.setFixedSize(34, 30)
        self.landmark_zoom_icon_btn.setToolTip("Zoom to the active marker; click again to return to Full view")
        self.landmark_zoom_icon_btn.clicked.connect(self._toggle_landmark_zoom)
        self.landmark_fullscreen_btn = QPushButton("⛶")
        self.landmark_fullscreen_btn.setFixedSize(34, 30)
        self.landmark_fullscreen_btn.setToolTip("Toggle full-screen window")
        self.landmark_fullscreen_btn.clicked.connect(self._toggle_landmark_fullscreen)
        for btn in (self.landmark_zoom_icon_btn, self.landmark_fullscreen_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton {border:1px solid #d1d5db; border-radius:6px; background:white; font-size:15px;}"
                "QPushButton:hover {background:#f3f4f6;}"
                "QPushButton:checked {background:#dbeafe; border-color:#60a5fa;}"
            )
        plot_tools.addWidget(self.landmark_zoom_icon_btn)
        plot_tools.addWidget(self.landmark_fullscreen_btn)
        plot_column.addLayout(plot_tools)

        self.landmarks_plot = LandmarksPlot()
        self.landmarks_plot.markerMoved.connect(self._on_landmark_plot_moved)
        plot_column.addWidget(self.landmarks_plot, 1)
        content.addLayout(plot_column, 1)

        panel = QFrame()
        panel.setFixedWidth(190)
        panel.setStyleSheet("QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(4)

        title = QLabel("MARKERS / LEGEND")
        title.setStyleSheet("border:none; font-size:10px; font-weight:800; color:#6b7280;")
        panel_layout.addWidget(title)

        self.landmark_marker_checks = {}
        self.landmark_marker_buttons = {}
        marker_specs = [
            ("spike_max", "Spike max", "#dc2626"),
            ("spike_min", "Spike min", "#991b1b"),
            ("spike_start", "Spike start", "#7e22ce"),
            ("t_wave_start", "T-wave start", "#2563eb"),
            ("t_wave_peak", "T-wave peak", "#1e3a8a"),
            ("t_wave_end", "T-wave end", "#0284c7"),
            ("EAD", "EAD", "#111827"),
        ]
        self._landmark_marker_specs = marker_specs
        self._landmark_marker_colors = {key: color for key, _, color in marker_specs}
        for key, label, color in marker_specs:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)

            cb = QCheckBox()
            cb.setCursor(Qt.PointingHandCursor)
            cb.setToolTip("Show/include this landmark")
            cb.setStyleSheet("QCheckBox {border:none; padding:1px;} QCheckBox::indicator {width:14px; height:14px;}")
            cb.toggled.connect(lambda checked, k=key: self._on_landmark_marker_toggled(k, checked))
            row.addWidget(cb, 0)

            marker_btn = QPushButton(label)
            marker_btn.setCheckable(True)
            marker_btn.setCursor(Qt.PointingHandCursor)
            marker_btn.setToolTip("Select this landmark for adjustment")
            marker_btn.setStyleSheet(
                f"QPushButton {{text-align:left; border:1px solid transparent; border-radius:5px; "
                f"padding:3px 5px; color:{color}; font-size:11px; font-weight:700; background:transparent;}}"
                "QPushButton:hover {background:#f3f4f6;}"
                "QPushButton:checked {background:#eef2ff; border-color:#93c5fd;}"
            )
            marker_btn.clicked.connect(lambda checked=False, k=key: self._on_landmark_marker_label_clicked(k))
            row.addWidget(marker_btn, 1)

            panel_layout.addWidget(row_widget)
            self.landmark_marker_checks[key] = cb
            self.landmark_marker_buttons[key] = marker_btn

        panel_layout.addSpacing(5)
        edit_label = QLabel("Adjusting marker")
        edit_label.setStyleSheet("border:none; font-size:10px; color:#4b5563;")
        panel_layout.addWidget(edit_label)
        self.landmark_selected_label = QLabel("Spike max")
        self.landmark_selected_label.setStyleSheet(
            "border:1px solid #dbeafe; border-radius:5px; background:#eff6ff; "
            "padding:4px 6px; font-size:11px; font-weight:700; color:#1f2937;"
        )
        panel_layout.addWidget(self.landmark_selected_label)

        time_label = QLabel("Selected marker time (ms)")
        time_label.setStyleSheet("border:none; font-size:10px; color:#4b5563;")
        panel_layout.addWidget(time_label)
        self.landmark_time_spin = QDoubleSpinBox()
        self.landmark_time_spin.setDecimals(4)
        self.landmark_time_spin.setRange(-1e12, 1e12)
        self.landmark_time_spin.setKeyboardTracking(False)
        self.landmark_time_spin.valueChanged.connect(self._on_landmark_spin_changed)
        panel_layout.addWidget(self.landmark_time_spin)

        panel_layout.addSpacing(4)
        view_title = QLabel("VIEW")
        view_title.setStyleSheet("border:none; font-size:10px; font-weight:800; color:#6b7280;")
        panel_layout.addWidget(view_title)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(4)
        self.landmark_full_view_btn = QPushButton("Full")
        self.landmark_selected_view_btn = QPushButton("Selected")
        for btn in (self.landmark_full_view_btn, self.landmark_selected_view_btn):
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton {border:1px solid #d1d5db; border-radius:6px; padding:5px; background:white; color:#374151;}"
                "QPushButton:hover {background:#f9fafb;}"
                "QPushButton:checked {background:#2563eb; color:white; border-color:#2563eb; font-weight:700;}"
            )
        self.landmark_view_group = QButtonGroup(self)
        self.landmark_view_group.setExclusive(True)
        self.landmark_view_group.addButton(self.landmark_full_view_btn)
        self.landmark_view_group.addButton(self.landmark_selected_view_btn)
        self.landmark_full_view_btn.clicked.connect(lambda checked=False: self._set_landmark_view_mode("full"))
        self.landmark_selected_view_btn.clicked.connect(lambda checked=False: self._set_landmark_view_mode("selected"))
        self.landmark_full_view_btn.setChecked(True)
        view_row.addWidget(self.landmark_full_view_btn)
        view_row.addWidget(self.landmark_selected_view_btn)
        panel_layout.addLayout(view_row)

        panel_layout.addSpacing(4)
        flags_title = QLabel("GROUP")
        flags_title.setStyleSheet("border:none; font-size:10px; font-weight:800; color:#6b7280;")
        panel_layout.addWidget(flags_title)
        self.exclude_spectrum_check = QCheckBox("Exclude entire spectrum")
        self.exclude_spectrum_check.setStyleSheet(
            "QCheckBox {border:none; font-size:10px; color:#374151;} QCheckBox::indicator {width:13px; height:13px;}"
        )
        self.exclude_spectrum_check.toggled.connect(self._on_exclude_spectrum_changed)
        panel_layout.addWidget(self.exclude_spectrum_check)

        reset_btn = QPushButton("Reset group to automatic")
        reset_btn.setToolTip(
            "Restore all v7 automatically calculated positions. Only Spike start, Spike max and Spike min are included by default."
        )
        reset_btn.clicked.connect(self._reset_current_landmarks)
        panel_layout.addWidget(reset_btn)

        self.landmark_noise_label = QLabel("Noise floor: —")
        self.landmark_noise_label.setWordWrap(True)
        self.landmark_noise_label.setStyleSheet("border:none; font-size:10px; color:#6b7280;")
        panel_layout.addWidget(self.landmark_noise_label)
        panel_layout.addStretch(1)
        content.addWidget(panel, 0)
        layout.addLayout(content, 1)

        group_row = QHBoxLayout()
        self.prev_landmark_btn = QPushButton("← Previous group")
        self.prev_landmark_btn.clicked.connect(lambda: self._change_landmark_group(-1))
        self.next_landmark_btn = QPushButton("Next group →")
        self.next_landmark_btn.clicked.connect(lambda: self._change_landmark_group(1))
        self.landmark_group_label = QLabel("No groups available")
        self.landmark_group_label.setAlignment(Qt.AlignCenter)
        self.landmark_group_label.setStyleSheet("font-size:12px; font-weight:700; color:#374151;")
        group_row.addWidget(self.prev_landmark_btn)
        group_row.addWidget(self.landmark_group_label, 1)
        group_row.addWidget(self.next_landmark_btn)
        layout.addLayout(group_row)

        bottom = QHBoxLayout()
        self.landmark_info = QLabel("Confirm Baseline first.")
        self.landmark_info.setStyleSheet("font-size:12px; color:#4b5563;")
        bottom.addWidget(self.landmark_info, 1)

        back_btn = QPushButton("← Baseline")
        back_btn.clicked.connect(lambda: self.navigate("baseline"))
        bottom.addWidget(back_btn)

        apply_btn = QPushButton("Use landmarks → Results")
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; "
            "padding:7px 16px; font-weight:700;}"
            "QPushButton:hover {background:#1d4ed8;}"
        )
        apply_btn.clicked.connect(self._confirm_landmarks_and_continue)
        bottom.addWidget(apply_btn)
        layout.addLayout(bottom)

        self._landmark_current_group = 1
        self._landmark_selected_marker = "spike_max"
        self._landmark_view_mode = "full"
        self._landmark_updating_controls = False
        return page

    @staticmethod
    def _default_landmark_enabled():
        return {
            "spike_max": True,
            "spike_min": True,
            "spike_start": True,
            "t_wave_start": False,
            "t_wave_peak": False,
            "t_wave_end": False,
            "EAD": False,
        }

    def _prepare_landmarks_page(self):
        if self.state.step_states["baseline"] != StepState.COMPLETE:
            QMessageBox.information(
                self,
                "Apply Baseline first",
                "Review the baseline for each group and confirm it before landmark adjustment.",
            )
            return False
        if not self.state.averaging_mean_snippets:
            QMessageBox.warning(self, "Landmarks", "No averaged mean waveforms are available for landmark extraction.")
            return False

        expected = set(self.state.averaging_mean_snippets.keys())
        stored = set(self.state.landmark_values.keys())
        if not self.state.landmark_values or stored != expected:
            if not self._calculate_landmarks_from_baselines():
                return False

        keys = sorted(self.state.landmark_values)
        if not keys:
            return False
        current = getattr(self, "_landmark_current_group", keys[0])
        self._landmark_current_group = current if current in keys else keys[0]
        self._landmark_view_mode = "full"
        self._show_landmark_group(self._landmark_current_group)
        self._update_landmark_info(applied=self.state.step_states["landmarks"] == StepState.COMPLETE)
        return True

    def _calculate_landmarks_from_baselines(self):
        values_by_group = {}
        auto_by_group = {}
        noise_by_group = {}
        flags_by_group = {}
        enabled_by_group = {}
        try:
            for group_idx in sorted(self.state.averaging_mean_snippets):
                time = self.state.averaging_snippet_times[group_idx]
                mean = self.state.averaging_mean_snippets[group_idx]
                baseline = self.state.baseline_values[group_idx]
                values, noise_floor = extract_landmarks_v7(
                    time,
                    mean,
                    baseline,
                    window_size_ms=50.0,
                    noise_floor_threshold=1.5,
                    t_wave_fallback_start_offset=30.0,
                )
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError(f"Group {group_idx} produced a non-finite landmark value.")
                values_by_group[int(group_idx)] = dict(values)
                auto_by_group[int(group_idx)] = dict(values)
                noise_by_group[int(group_idx)] = float(noise_floor)
                flags_by_group[int(group_idx)] = {"exclude_spectrum": False}
                enabled_by_group[int(group_idx)] = self._default_landmark_enabled()
        except Exception as exc:
            QMessageBox.warning(self, "Landmarks", str(exc))
            return False

        self.state.landmark_values = values_by_group
        self.state.landmark_auto_values = auto_by_group
        self.state.landmark_noise_floor = noise_by_group
        self.state.landmark_flags = flags_by_group
        self.state.landmark_enabled = enabled_by_group
        self._landmark_current_group = min(values_by_group)
        self._landmark_selected_marker = "spike_max"
        self._landmark_view_mode = "full"
        return True

    def _show_landmark_group(self, group_idx):
        keys = sorted(self.state.landmark_values)
        if not keys:
            self.landmark_group_label.setText("No groups available")
            return
        if group_idx not in keys:
            group_idx = keys[0]
        self._landmark_current_group = int(group_idx)

        time = np.asarray(self.state.averaging_snippet_times[group_idx], dtype=float)
        mean = self.state.averaging_mean_snippets[group_idx]
        baseline = self.state.baseline_values[group_idx]
        values = self.state.landmark_values[group_idx]
        flags = self.state.landmark_flags.get(group_idx, {"exclude_spectrum": False})
        enabled = self.state.landmark_enabled.setdefault(group_idx, self._default_landmark_enabled())

        selected = getattr(self, "_landmark_selected_marker", "spike_max")
        if selected not in values or not enabled.get(selected, False):
            selected = next((k for k, _, _ in self._landmark_marker_specs if enabled.get(k, False)), "spike_max")
            self._landmark_selected_marker = selected

        self.landmarks_plot.set_group(group_idx, time, mean, baseline, values, flags, enabled, selected)

        self._landmark_updating_controls = True
        try:
            for key, cb in self.landmark_marker_checks.items():
                cb.setChecked(bool(enabled.get(key, False)))
            for key, btn in self.landmark_marker_buttons.items():
                btn.setChecked(key == selected)
            selected_label = next((label for key, label, _ in self._landmark_marker_specs if key == selected), selected)
            self.landmark_selected_label.setText(selected_label)
            finite_time = time[np.isfinite(time)]
            if finite_time.size:
                self.landmark_time_spin.setRange(float(np.min(finite_time)), float(np.max(finite_time)))
            self.landmark_time_spin.setValue(float(values[selected]))
            self.landmark_time_spin.setEnabled(bool(enabled.get(selected, False)))
            self.exclude_spectrum_check.setChecked(bool(flags.get("exclude_spectrum", False)))
        finally:
            self._landmark_updating_controls = False

        position = keys.index(group_idx) + 1
        self.landmark_group_label.setText(f"Group {group_idx} ({position}/{len(keys)})")
        self.prev_landmark_btn.setEnabled(position > 1)
        self.next_landmark_btn.setEnabled(position < len(keys))
        noise = self.state.landmark_noise_floor.get(group_idx)
        self.landmark_noise_label.setText(
            "Noise floor: —" if noise is None else f"v7 noise floor: {noise:.4g} mV  •  threshold: 1.5σ"
        )
        # Keep the highlighted view button and the actual x-limits synchronized.
        # Full is the default requested view when entering Landmarks.
        mode = getattr(self, "_landmark_view_mode", "full")
        self._landmark_updating_controls = True
        try:
            self.landmark_full_view_btn.setChecked(mode == "full")
            self.landmark_selected_view_btn.setChecked(mode == "selected")
            self.landmark_zoom_icon_btn.blockSignals(True)
            self.landmark_zoom_icon_btn.setChecked(mode == "selected")
            self.landmark_zoom_icon_btn.blockSignals(False)
        finally:
            self._landmark_updating_controls = False
        self._apply_landmark_view_mode()

    def _change_landmark_group(self, direction):
        keys = sorted(self.state.landmark_values)
        if not keys:
            return
        try:
            pos = keys.index(self._landmark_current_group)
        except ValueError:
            pos = 0
        pos = max(0, min(len(keys) - 1, pos + int(direction)))
        self._show_landmark_group(keys[pos])

    def _select_landmark_for_edit(self, marker_key):
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_values:
            return
        enabled = self.state.landmark_enabled.get(group_idx, {})
        if not enabled.get(marker_key, False):
            return
        self._landmark_selected_marker = marker_key
        self._landmark_updating_controls = True
        try:
            for key, btn in self.landmark_marker_buttons.items():
                btn.setChecked(key == marker_key)
            selected_label = next((label for key, label, _ in self._landmark_marker_specs if key == marker_key), marker_key)
            self.landmark_selected_label.setText(selected_label)
            self.landmark_time_spin.setValue(float(self.state.landmark_values[group_idx][marker_key]))
            self.landmark_time_spin.setEnabled(True)
        finally:
            self._landmark_updating_controls = False
        self.landmarks_plot.set_selected_marker(marker_key)
        if getattr(self, "_landmark_view_mode", "full") == "selected":
            self.landmarks_plot.focus_selected()

    def _on_landmark_marker_toggled(self, marker_key, checked):
        if getattr(self, "_landmark_updating_controls", False):
            return
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_values:
            return
        enabled = self.state.landmark_enabled.setdefault(group_idx, self._default_landmark_enabled())
        enabled[marker_key] = bool(checked)
        self.landmarks_plot.set_enabled(enabled)

        if checked:
            self._select_landmark_for_edit(marker_key)
        elif marker_key == getattr(self, "_landmark_selected_marker", None):
            next_key = next((k for k, _, _ in self._landmark_marker_specs if enabled.get(k, False)), None)
            if next_key:
                self._select_landmark_for_edit(next_key)
            else:
                self.landmark_time_spin.setEnabled(False)

        self.state.mark_step_dirty("landmarks")
        self._update_landmark_info(applied=False)
        self.refresh_all()

    def _on_landmark_marker_label_clicked(self, marker_key):
        """Select a visible legend item as the marker being adjusted.

        The checkbox remains responsible for inclusion/visibility, while the
        colored legend name replaces the old Edit marker dropdown.
        """
        if getattr(self, "_landmark_updating_controls", False):
            return
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_values:
            return
        enabled = self.state.landmark_enabled.setdefault(group_idx, self._default_landmark_enabled())
        if not enabled.get(marker_key, False):
            # Selecting a hidden marker also makes it visible/included while
            # preserving its existing automatic/manual value.
            self._landmark_updating_controls = True
            self.landmark_marker_checks[marker_key].setChecked(True)
            self._landmark_updating_controls = False
            enabled[marker_key] = True
            self.landmarks_plot.set_enabled(enabled)
            self.state.mark_step_dirty("landmarks")
        self._select_landmark_for_edit(marker_key)
        self._update_landmark_info(applied=False)
        self.refresh_all()

    def _toggle_landmark_zoom(self, checked=False):
        self._set_landmark_view_mode("selected" if bool(checked) else "full")

    def _toggle_landmark_fullscreen(self):
        if not self._landmark_fullscreen_active:
            self._landmark_was_maximized = self.isMaximized()
            self._landmark_fullscreen_active = True
            self.showFullScreen()
            self.landmark_fullscreen_btn.setToolTip("Exit full-screen window")
        else:
            self._landmark_fullscreen_active = False
            if self._landmark_was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
            self.landmark_fullscreen_btn.setToolTip("Toggle full-screen window")

    def _set_landmark_view_mode(self, mode):
        self._landmark_view_mode = "full" if mode == "full" else "selected"
        self._landmark_updating_controls = True
        try:
            self.landmark_full_view_btn.setChecked(self._landmark_view_mode == "full")
            self.landmark_selected_view_btn.setChecked(self._landmark_view_mode == "selected")
            self.landmark_zoom_icon_btn.blockSignals(True)
            self.landmark_zoom_icon_btn.setChecked(self._landmark_view_mode == "selected")
            self.landmark_zoom_icon_btn.blockSignals(False)
        finally:
            self._landmark_updating_controls = False
        self._apply_landmark_view_mode()

    def _apply_landmark_view_mode(self):
        if getattr(self, "_landmark_view_mode", "full") == "full":
            self.landmarks_plot.reset_view()
        else:
            self.landmarks_plot.focus_selected()

    def _set_landmark_value(self, marker_key, value):
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_values or marker_key not in self.state.landmark_values[group_idx]:
            return
        if not self.state.landmark_enabled.get(group_idx, {}).get(marker_key, False):
            return
        if not np.isfinite(value):
            return
        self.state.landmark_values[group_idx][marker_key] = float(value)
        self.landmarks_plot.set_marker_value(marker_key, float(value), redraw=False)
        if marker_key == getattr(self, "_landmark_selected_marker", None):
            self._landmark_updating_controls = True
            self.landmark_time_spin.setValue(float(value))
            self._landmark_updating_controls = False
        self.landmarks_plot.canvas.draw_idle()
        self.state.mark_step_dirty("landmarks")
        self._update_landmark_info(applied=False)
        self.refresh_all()

    def _on_landmark_plot_moved(self, marker_key, value):
        self._set_landmark_value(marker_key, float(value))

    def _on_landmark_spin_changed(self, value):
        if getattr(self, "_landmark_updating_controls", False):
            return
        self._set_landmark_value(getattr(self, "_landmark_selected_marker", "spike_max"), float(value))

    def _on_exclude_spectrum_changed(self):
        if getattr(self, "_landmark_updating_controls", False):
            return
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_values:
            return
        flags = self.state.landmark_flags.setdefault(group_idx, {"exclude_spectrum": False})
        flags["exclude_spectrum"] = bool(self.exclude_spectrum_check.isChecked())
        self.landmarks_plot.set_flags(flags)
        self.state.mark_step_dirty("landmarks")
        self._update_landmark_info(applied=False)
        self.refresh_all()

    def _reset_current_landmarks(self):
        group_idx = getattr(self, "_landmark_current_group", None)
        if group_idx not in self.state.landmark_auto_values:
            return
        self.state.landmark_values[group_idx] = dict(self.state.landmark_auto_values[group_idx])
        self.state.landmark_flags[group_idx] = {"exclude_spectrum": False}
        self.state.landmark_enabled[group_idx] = self._default_landmark_enabled()
        self._landmark_selected_marker = "spike_max"
        self.state.mark_step_dirty("landmarks")
        self._show_landmark_group(group_idx)
        self._update_landmark_info(applied=False)
        self.refresh_all()

    def _update_landmark_info(self, applied=False):
        count = len(self.state.landmark_values)
        ead_count = sum(1 for enabled in self.state.landmark_enabled.values() if enabled.get("EAD", False))
        optional_on = sum(
            sum(1 for key in ("t_wave_start", "t_wave_peak", "t_wave_end", "EAD") if enabled.get(key, False))
            for enabled in self.state.landmark_enabled.values()
        )
        excluded = sum(1 for flags in self.state.landmark_flags.values() if flags.get("exclude_spectrum"))
        status = "Applied" if applied else "Ready"
        self.landmark_info.setStyleSheet(
            "font-size:12px; color:#166534;" if applied else "font-size:12px; color:#4b5563;"
        )
        self.landmark_info.setText(
            f"{status}  •  {count} group(s)  •  optional markers included: {optional_on}  •  EAD: {ead_count}  •  excluded groups: {excluded}"
        )

    def _apply_landmarks(self):
        expected = set(self.state.averaging_mean_snippets.keys())
        available = set(self.state.landmark_values.keys())
        if not expected or available != expected:
            QMessageBox.warning(self, "Landmarks", "Landmark values are not available for every averaging group.")
            return
        bad = []
        required = {"spike_max", "spike_min", "spike_start", "t_wave_start", "t_wave_peak", "t_wave_end", "EAD"}
        for group_idx, values in self.state.landmark_values.items():
            if not required.issubset(values) or not all(np.isfinite(values[k]) for k in required):
                bad.append(group_idx)
        if bad:
            QMessageBox.warning(
                self,
                "Landmarks",
                f"Invalid landmark value(s) in group(s): {', '.join(map(str, bad))}.",
            )
            return
        self.state.invalidate_from("landmarks")
        self._update_landmark_info(applied=True)
        self._show_landmark_group(self._landmark_current_group)
        self.refresh_all()

    def _confirm_landmarks_and_continue(self):
        self._apply_landmarks()
        if self.state.step_states["landmarks"] == StepState.COMPLETE:
            self.navigate("results")

    def _build_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        intro = QLabel(
            "Results use the v7 FPD, Fridericia and Bazett formulas. Only landmark markers that are checked/included "
            "contribute to metrics. For example, FPD_New requires Spike start + T-wave end, while FPD_Traditional "
            "requires Spike start + T-wave peak."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size:11px; color:#4b5563;")
        layout.addWidget(intro)

        self.results_summary = QLabel("Confirm Landmarks first.")
        self.results_summary.setStyleSheet(
            "QLabel {background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:7px; "
            "font-size:11px; color:#374151;}"
        )
        layout.addWidget(self.results_summary)

        self.results_table = QTableWidget(0, 10)
        self.results_table.setHorizontalHeaderLabels([
            "Group", "ISI mean", "ISI SD", "FPD New", "FPD Traditional",
            "New Fridericia", "Traditional Fridericia", "New Bazett", "Traditional Bazett", "EAD time",
        ])
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, 1)

        self.export_status = QLabel("Review the table, then save the compact Summary workbook and PDF report.")
        self.export_status.setStyleSheet("font-size:11px; color:#4b5563;")
        layout.addWidget(self.export_status)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setVisible(False)
        layout.addWidget(self.export_progress)

        # Save destination lives here, immediately above the final save/next-file
        # controls, so it can be changed at the moment of export.
        save_config_row = QHBoxLayout()
        save_config_row.addStretch(1)
        save_config = QFrame()
        save_config.setMaximumWidth(560)
        save_config.setStyleSheet(
            "QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:9px;}"
        )
        save_layout = QVBoxLayout(save_config)
        save_layout.setContentsMargins(10, 8, 10, 8)
        save_layout.setSpacing(5)

        save_title = QLabel("SAVE RESULTS")
        save_title.setStyleSheet("border:none; font-size:10px; font-weight:800; color:#6b7280;")
        save_layout.addWidget(save_title)

        name_row = QHBoxLayout()
        name_label = QLabel("Analysis name")
        name_label.setFixedWidth(82)
        name_label.setStyleSheet("border:none; font-size:10px; color:#4b5563;")
        self.analysis_name_edit = QLineEdit()
        self.analysis_name_edit.textChanged.connect(self._output_changed)
        name_row.addWidget(name_label)
        name_row.addWidget(self.analysis_name_edit, 1)
        save_layout.addLayout(name_row)

        folder_row = QHBoxLayout()
        folder_label = QLabel("Save folder")
        folder_label.setFixedWidth(82)
        folder_label.setStyleSheet("border:none; font-size:10px; color:#4b5563;")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(self._output_changed)
        browse_output = QPushButton("Browse…")
        browse_output.clicked.connect(self.choose_output_dir)
        folder_row.addWidget(folder_label)
        folder_row.addWidget(self.output_dir_edit, 1)
        folder_row.addWidget(browse_output)
        save_layout.addLayout(folder_row)

        self.output_preview = QLabel("")
        self.output_preview.setStyleSheet("border:none; font-size:9px; color:#6b7280;")
        self.output_preview.setWordWrap(True)
        save_layout.addWidget(self.output_preview)
        save_config_row.addWidget(save_config)
        layout.addLayout(save_config_row)

        bottom = QHBoxLayout()
        back_btn = QPushButton("← Landmarks")
        back_btn.clicked.connect(lambda: self.navigate("landmarks"))
        bottom.addWidget(back_btn)
        bottom.addStretch(1)

        self.save_results_btn = QPushButton("Save results")
        self.save_results_btn.setMinimumHeight(38)
        self.save_results_btn.setStyleSheet(
            "QPushButton {background:#2563eb; color:white; border:none; border-radius:7px; padding:8px 18px; font-weight:800;}"
            "QPushButton:hover {background:#1d4ed8;}"
            "QPushButton:disabled {background:#d1d5db; color:#6b7280;}"
        )
        self.save_results_btn.clicked.connect(self._save_results)
        bottom.addWidget(self.save_results_btn)

        self.next_file_btn = QPushButton("Start next file")
        self.next_file_btn.setMinimumHeight(42)
        self.next_file_btn.setEnabled(False)
        self.next_file_btn.setStyleSheet(
            "QPushButton {background:#059669; color:white; border:none; border-radius:8px; padding:9px 20px; font-weight:800;}"
            "QPushButton:hover {background:#047857;}"
            "QPushButton:disabled {background:#e5e7eb; color:#9ca3af;}"
        )
        self.next_file_btn.clicked.connect(lambda: self._restart_analysis(force=True))
        bottom.addWidget(self.next_file_btn)
        layout.addLayout(bottom)
        return page

    def _prepare_results_page(self):
        if self.state.step_states["landmarks"] != StepState.COMPLETE:
            return False
        try:
            by_group, overall, ead_count = compute_results_v7(
                self.state.landmark_values,
                self.state.landmark_enabled,
                self.state.landmark_flags,
                self.state.averaging_isi_results,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Results", str(exc))
            return False
        self.state.results_by_group = by_group
        self.state.results_overall = overall
        self.state.results_ead_count = ead_count
        self._populate_results_table()
        self._update_results_controls()
        return True

    @staticmethod
    def _format_result_value(value):
        if value is None:
            return "—"
        try:
            value = float(value)
        except Exception:
            return str(value)
        return "—" if not np.isfinite(value) else f"{value:.4f}"

    def _populate_results_table(self):
        groups = sorted(self.state.results_by_group)
        self.results_table.setRowCount(len(groups))
        keys = [
            "isi_mean", "isi_std", "fpd_new", "fpd_traditional",
            "fpd_new_fridericia", "fpd_traditional_fridericia",
            "fpd_new_bazett", "fpd_traditional_bazett", "ead_time",
        ]
        for row, group_idx in enumerate(groups):
            metrics = self.state.results_by_group[group_idx]
            self.results_table.setItem(row, 0, QTableWidgetItem(f"B{group_idx:02d}"))
            for col, key in enumerate(keys, start=1):
                item = QTableWidgetItem(self._format_result_value(metrics.get(key)))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.results_table.setItem(row, col, item)

        fpd_new_mean = self.state.results_overall.get("fpd_new", (None, None))[0]
        fpd_trad_mean = self.state.results_overall.get("fpd_traditional", (None, None))[0]
        self.results_summary.setText(
            f"Groups: {len(groups)}  •  EAD included: {self.state.results_ead_count}  •  "
            f"Mean FPD New: {self._format_result_value(fpd_new_mean)}  •  "
            f"Mean FPD Traditional: {self._format_result_value(fpd_trad_mean)}"
        )

    def _update_results_controls(self):
        exported = bool(self.state.results_exported)
        self.next_file_btn.setEnabled(exported)
        if exported and self.state.results_excel_path and self.state.results_pdf_path:
            self.export_status.setStyleSheet("font-size:11px; color:#166534; font-weight:600;")
            self.export_status.setText(
                f"Saved: {self.state.results_excel_path.name} and {self.state.results_pdf_path.name}. You can now start the next file."
            )
        else:
            self.export_status.setStyleSheet("font-size:11px; color:#4b5563;")
            self.export_status.setText("Review the table, then save the compact Summary workbook and PDF report.")
        self._update_restart_controls()

    def _save_results(self):
        if not self.state.results_by_group:
            if not self._prepare_results_page():
                return
        name = (self.analysis_name_edit.text().strip() or self.state.analysis_name or "analysis").strip()
        output_text = self.output_dir_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Output folder", "Choose an output folder on the Results page before saving.")
            return
        output_dir = Path(output_text)
        excel_path = output_dir / f"{name}.xlsx"
        pdf_path = output_dir / f"{name}.pdf"

        self.save_results_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)
        self.export_status.setText("Saving results...")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def progress(pct, message):
            self.export_progress.setValue(int(pct))
            self.export_status.setText(str(message))
            QApplication.processEvents()

        try:
            # Export dependencies are lazy/background-preloaded so startup stays lighter.
            from .exporter import export_results
            export_results(self.state, excel_path, pdf_path, progress)
        except Exception as exc:
            logging.getLogger(__name__).exception("Results export failed")
            for partial in (excel_path, pdf_path):
                try:
                    if partial.exists():
                        partial.unlink()
                except Exception:
                    pass
            QMessageBox.critical(self, "Save results", f"Could not save results:\n{exc}")
            self.export_status.setText("Save failed. Nothing was marked complete.")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.save_results_btn.setEnabled(True)

        self.state.results_exported = True
        self.state.results_excel_path = excel_path
        self.state.results_pdf_path = pdf_path
        self.state.step_states["results"] = StepState.COMPLETE
        self.export_progress.setValue(100)
        self._update_results_controls()
        self.refresh_all()
        QMessageBox.information(
            self,
            "Results saved",
            f"Excel and PDF saved to:\n{output_dir}",
        )

    def _update_restart_controls(self):
        on_results = self.pages.get("results") is not None and self.stack.currentWidget() is self.pages.get("results")
        enabled = bool(self.state.recording.source_path)
        if on_results and not self.state.results_exported:
            enabled = False
        self.restart_btn.setEnabled(enabled)
        if hasattr(self, "next_file_btn"):
            self.next_file_btn.setEnabled(bool(self.state.results_exported))

    def _restart_analysis(self, checked=False, force=False):
        if not self.state.recording.source_path:
            return
        on_results = self.pages.get("results") is not None and self.stack.currentWidget() is self.pages.get("results")
        if on_results and not self.state.results_exported and not force:
            return
        if not force and not self.state.results_exported:
            answer = QMessageBox.question(
                self,
                "Restart analysis",
                "Clear the current unsaved analysis and return to Browse?",
            )
            if answer != QMessageBox.Yes:
                return

        # Reset in place rather than replacing the AnalysisState object. Long-lived
        # pages and callbacks are reused across analyses; preserving object identity
        # avoids stale references when Start next file is used.
        self.state.reset_all()
        self.selected_preview_electrode = None
        self._fast_electrode_requested = None
        self._fast_electrode_failed.clear()
        self._range_preview_requested = None
        self._range_preview_failed.clear()
        self.range_plot.clear_signal()
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setRange(0.0, 1.0)
        self.range_end_spin.setRange(0.0, 1.0)
        self.range_start_spin.setValue(0.0)
        self.range_end_spin.setValue(1.0)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)
        self.range_info.setText("Select an electrode, then choose the interval to analyze.")
        self.range_info.setStyleSheet("font-size:12px; color:#4b5563;")
        self.recording_path.clear()
        self.loading_status.setText("Select a recording to begin.")
        self.loading_progress.setValue(0)
        self.recording_details.clear()
        self.analysis_name_edit.blockSignals(True)
        self.output_dir_edit.blockSignals(True)
        self.analysis_name_edit.clear()
        self.output_dir_edit.clear()
        self.analysis_name_edit.blockSignals(False)
        self.output_dir_edit.blockSignals(False)
        self.populate_electrode_grid()
        self.results_table.setRowCount(0)
        self.results_summary.setText("Confirm Landmarks first.")
        self.export_progress.setVisible(False)
        self.export_progress.setValue(0)
        self.export_status.setText("Review the table, then save the compact Summary workbook and PDF report.")
        self.next_file_btn.setEnabled(False)
        self.refresh_all()
        self.navigate("recording")

    def _build_placeholder_page(self, step):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        card = QFrame()
        card.setStyleSheet("QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px;}")
        c = QVBoxLayout(card)
        title = QLabel(f"{step.title()} — next build stage")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:700; color:#374151; border:none;")
        subtitle = QLabel(
            "This stage will use the frozen v7 scientific logic without changing the calculation method.\n"
            "Range selection is now active; this page is the next stage to be implemented."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size:14px; color:#6b7280; border:none;")
        c.addStretch(1)
        c.addWidget(title)
        c.addWidget(subtitle)
        c.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def navigate(self, key):
        if key != "landmarks" and getattr(self, "_landmark_fullscreen_active", False):
            self._toggle_landmark_fullscreen()
        if key == "electrode" and not self.state.recording.electrode_columns:
            QMessageBox.information(self, "Select a recording", "Open a recording first.")
            key = "recording"
        if key in ANALYSIS_STEPS and not self.state.selected_electrode:
            QMessageBox.information(self, "Select an electrode", "Choose the recording and electrode first.")
            key = "electrode" if self.state.recording.electrode_columns else "recording"

        # Earlier analysis stages are always revisitable. This is intentional:
        # users must be able to go back, change Range/Detection, and let the
        # dependency system mark downstream work as needing update.
        if key == "range" and self.state.selected_electrode:
            if not self._prepare_range_page():
                key = "electrode"
        elif key == "detection" and self.state.selected_electrode:
            if not self._prepare_detection_page():
                key = "range"
                self._prepare_range_page()
        elif key == "averaging" and self.state.selected_electrode:
            if not self._prepare_averaging_page():
                key = "detection"
                self._prepare_detection_page()
        elif key == "baseline" and self.state.selected_electrode:
            if self.state.step_states["averaging"] != StepState.COMPLETE:
                QMessageBox.information(
                    self,
                    "Apply Averaging first",
                    "Review the averaging groups and confirm them before baseline selection.",
                )
                key = "averaging"
                if not self._prepare_averaging_page():
                    key = "detection"
                    self._prepare_detection_page()
            elif not self._prepare_baseline_page():
                key = "averaging"
                self._prepare_averaging_page()
        elif key == "landmarks":
            if self.state.step_states["baseline"] != StepState.COMPLETE:
                QMessageBox.information(
                    self,
                    "Apply Baseline first",
                    "Review the baseline for each group and confirm it before continuing.",
                )
                key = "baseline"
                if not self._prepare_baseline_page():
                    key = "averaging"
                    self._prepare_averaging_page()
            elif not self._prepare_landmarks_page():
                key = "baseline"
                self._prepare_baseline_page()
        elif key == "results":
            if self.state.step_states["baseline"] != StepState.COMPLETE:
                QMessageBox.information(
                    self,
                    "Apply Baseline first",
                    "Review the baseline for each group and confirm it before continuing.",
                )
                key = "baseline"
                self._prepare_baseline_page()
            elif self.state.step_states["landmarks"] != StepState.COMPLETE:
                QMessageBox.information(
                    self,
                    "Apply Landmarks first",
                    "Review the landmark positions and included markers for each group and confirm them before Results.",
                )
                key = "landmarks"
                if not self._prepare_landmarks_page():
                    key = "baseline"
                    self._prepare_baseline_page()
            elif not self._prepare_results_page():
                key = "landmarks"
                self._prepare_landmarks_page()
        for k, button in self.nav_buttons.items():
            button.setChecked(k == key)
        self.page_heading.setText(self._heading_for(key))
        self.stack.setCurrentWidget(self.pages[key])
        if key == "cache":
            self.refresh_cache_page()
        self._update_restart_controls()

    def _heading_for(self, key):
        labels = {
            "recording": "Recording",
            "electrode": "Electrode selection",
            "cache": "Cache",
        }
        return labels.get(key, key.title())

    def choose_recording(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MEA recording",
            "",
            "MEA recordings (*.csv *.CSV *.tsv *.txt *.xlsx);;CSV files (*.csv *.CSV);;Excel files (*.xlsx);;All files (*)",
        )
        if not path:
            return
        source = Path(path)
        self.recording_path.setText(str(source))
        self.loading_progress.setValue(0)
        self.loading_status.setText("Preparing recording…")
        self.recording_details.setText("")
        self._set_busy(True)

        thread = QThread(self)
        worker = RecordingLoadWorker(source, self.cache_manager)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_load_progress)
        worker.finished.connect(self._on_recording_loaded)
        worker.failed.connect(self._on_recording_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "_worker_thread", None))
        self._worker_thread = thread
        self._worker = worker
        thread.finished.connect(lambda: setattr(self, "_worker", None))
        thread.start()

    def _set_busy(self, busy: bool):
        self.nav.setEnabled(not busy)
        self.loading_progress.setEnabled(True)

    def _start_background_preload(self):
        if self._preload_thread is not None:
            return
        # Range responsiveness has priority. For auto-selected single-electrode
        # recordings, let the fast electrode sidecar finish before warming less
        # urgent SciPy/export modules so the two jobs do not compete for disk/CPU.
        if self._fast_electrode_thread is not None:
            QTimer.singleShot(500, self._start_background_preload)
            return
        thread = QThread(self)
        worker = BackgroundPreloadWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._background_preload_finished)
        self._preload_thread = thread
        self._preload_worker = worker
        thread.start()

    def _background_preload_finished(self):
        self._preload_thread = None
        self._preload_worker = None

    def _on_load_progress(self, value, message):
        self.loading_progress.setValue(value)
        self.loading_status.setText(message)

    def _on_recording_loaded(self, recording):
        # A Recording page can be reused for multiple files in one app session.
        # Clear Range's old Matplotlib artists before installing the new recording.
        self.range_plot.clear_signal()
        self.state.recording = recording
        self.state.reset_for_new_recording()
        self._fast_electrode_requested = None
        self._fast_electrode_failed.clear()
        self.loading_progress.setValue(100)
        self.loading_status.setText("Recording ready")
        self._refresh_recording_details()
        stem = recording.source_path.stem
        default_dir = recording.source_path.parent / f"{stem}_results"
        self.analysis_name_edit.blockSignals(True)
        self.output_dir_edit.blockSignals(True)
        self.analysis_name_edit.setText(f"{stem}_analysis")
        self.output_dir_edit.setText(str(default_dir))
        self.analysis_name_edit.blockSignals(False)
        self.output_dir_edit.blockSignals(False)
        self.state.analysis_name = self.analysis_name_edit.text().strip()
        self.state.output_dir = Path(self.output_dir_edit.text())
        self.populate_electrode_grid()
        self._set_busy(False)
        self.refresh_cache_page()
        self.refresh_all()
        # For lazy multi-GB CSVs, keep CPU/disk bandwidth free for the selected
        # electrode extraction. Warm optional libraries after that scan finishes.
        if not (recording.source_format.casefold() == "csv" and recording.cache_parquet is None):
            QTimer.singleShot(150, self._start_background_preload)

        usable_electrodes = [
            e for e in recording.electrode_columns
            if recording.electrode_has_data.get(e, True)
        ]
        if len(recording.electrode_columns) == 1 and usable_electrodes:
            only = recording.electrode_columns[0]
            self.selected_preview_electrode = only
            self._show_large_preview(only)
            self.confirm_electrode()
        else:
            self.navigate("electrode")

    def _on_recording_failed(self, details):
        self._set_busy(False)
        self.loading_status.setText("Could not open recording")
        self.loading_progress.setValue(0)
        QMessageBox.critical(self, "Recording error", details)

    def populate_electrode_grid(self):
        while self.electrode_grid.count():
            item = self.electrode_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.electrode_cards.clear()
        rec = self.state.recording
        for idx, electrode in enumerate(rec.electrode_columns):
            usable = rec.electrode_has_data.get(electrode, True)
            card = ElectrodeCard(
                electrode,
                rec.preview_time,
                rec.preview_signals.get(electrode),
                usable=usable,
            )
            card.clicked.connect(self.select_preview_electrode)
            self.electrode_cards[electrode] = card
            # Larger mini-plots: five across, vertically scrollable.
            self.electrode_grid.addWidget(card, idx // 5, idx % 5)
        self.selected_preview_electrode = None
        self.use_electrode_btn.setEnabled(False)
        self.large_preview.title.setText("Select an electrode")
        self.large_preview.spark.set_data([], [])
        self.large_preview.meta.setText("")

    def select_preview_electrode(self, electrode):
        rec = self.state.recording
        if not rec.electrode_has_data.get(electrode, True):
            return
        self.selected_preview_electrode = electrode
        for name, card in self.electrode_cards.items():
            card.set_selected(name == electrode)
        self._show_large_preview(electrode)
        self.use_electrode_btn.setEnabled(True)

    def _show_large_preview(self, electrode):
        rec = self.state.recording
        self.large_preview.set_signal(
            electrode,
            rec.preview_time,
            rec.preview_signals.get(electrode),
            rec.row_count,
            rec.row_count_exact,
        )

    def _refresh_recording_details(self):
        recording = self.state.recording
        if recording.source_path is None:
            self.recording_details.setText("")
            return
        format_label = (
            recording.source_format
            or recording.source_path.suffix.lstrip(".")
            or "recording"
        ).upper()
        row_label = (
            f"{recording.row_count:,}"
            if recording.row_count_exact
            else f"~{recording.row_count:,} estimated"
        )
        self.recording_details.setText(
            f"{format_label}  •  {row_label} data rows  •  "
            f"{len(recording.electrode_columns)} electrode(s)  •  "
            f"{human_size(recording.source_size)} source file"
        )

    def confirm_electrode(self):
        electrode = self.selected_preview_electrode
        if not electrode:
            return
        changed = electrode != self.state.selected_electrode
        self.state.selected_electrode = electrode
        if changed:
            # Keep user settings visible, but full-resolution range data belong
            # to the old electrode and must be loaded again after Range is applied.
            self.state.range_time = None
            self.state.range_voltage_raw = None
            # When the scientific pages are active, changing the electrode invalidates all downstream work.
            for step in ANALYSIS_STEPS:
                if self.state.step_states[step] == StepState.COMPLETE:
                    self.state.step_states[step] = StepState.NEEDS_UPDATE
                elif self.state.step_states[step] != StepState.SKIPPED:
                    self.state.step_states[step] = StepState.NOT_STARTED
        self.refresh_all()
        self.navigate("range")
        if (
            self.state.recording.source_format.casefold() == "csv"
            and self.state.recording.cache_parquet is None
        ):
            # Build 20: do not scan/extract the entire selected electrode. Build
            # only a lightweight ~30k-point Range preview in the background.
            self._start_range_preview_preparation(electrode)
        else:
            # XLSX / Parquet recordings retain the established fast-sidecar path.
            self._start_fast_electrode_preparation(electrode)

    def _start_range_preview_preparation(self, electrode: str | None):
        if not electrode or electrode not in self.state.recording.electrode_columns:
            return
        if self.state.recording.source_format.casefold() != "csv" or self.state.recording.cache_parquet is not None:
            return
        if load_range_preview(self.state.recording, electrode) is not None:
            self._range_preview_failed.discard(electrode)
            self._update_range_fast_access_ui()
            return

        self._range_preview_failed.discard(electrode)
        self._range_preview_requested = electrode
        if self._range_preview_thread is not None:
            return

        self._range_preview_electrode = electrode
        self._range_preview_progress = 0
        thread = QThread(self)
        worker = RangePreviewWorker(self.state.recording, electrode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_range_preview_progress)
        worker.finished.connect(self._on_range_preview_ready)
        worker.failed.connect(self._on_range_preview_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._range_preview_thread_finished)
        self._range_preview_thread = thread
        self._range_preview_worker = worker
        thread.start()

    def _on_range_preview_progress(self, electrode: str, pct: int):
        if electrode == self._range_preview_electrode:
            self._range_preview_progress = int(pct)

    def _on_range_preview_ready(self, electrode: str):
        self._range_preview_failed.discard(electrode)
        if electrode == self.state.selected_electrode:
            enhanced = load_range_preview(self.state.recording, electrode)
            if enhanced is not None and self.range_plot.selection is not None:
                start, end = self.range_plot.selection
                full_start = (
                    float(self.state.recording.time_min)
                    if self.state.recording.time_min is not None
                    else float(np.nanmin(enhanced[0]))
                )
                full_end = (
                    float(self.state.recording.time_max)
                    if self.state.recording.time_max is not None
                    else float(np.nanmax(enhanced[0]))
                )
                self.range_plot.set_signal(
                    enhanced[0], enhanced[1], start, end,
                    full_start=full_start, full_end=full_end,
                )
        self._update_range_fast_access_ui()

    def _on_range_preview_failed(self, electrode: str, details: str):
        self._range_preview_failed.add(electrode)
        self._update_range_fast_access_ui()
        logging.getLogger(__name__).warning(
            "Enhanced Range preview failed for %s; retaining 5k preview. %s",
            electrode, details,
        )

    def _range_preview_thread_finished(self):
        finished = self._range_preview_electrode
        self._range_preview_thread = None
        self._range_preview_worker = None
        self._range_preview_electrode = None
        self._range_preview_progress = 0
        requested = self._range_preview_requested
        self._update_range_fast_access_ui()
        if requested and requested != finished and requested == self.state.selected_electrode:
            QTimer.singleShot(0, lambda e=requested: self._start_range_preview_preparation(e))

    def _start_fast_electrode_preparation(self, electrode: str | None):
        if not electrode or electrode not in self.state.recording.electrode_columns:
            return
        if fast_electrode_cache_ready(self.state.recording, electrode):
            self._fast_electrode_failed.discard(electrode)
            self._update_range_fast_access_ui()
            return

        self._fast_electrode_failed.discard(electrode)
        self._fast_electrode_requested = electrode
        if self._fast_electrode_thread is not None:
            # Finish the currently running single-column job, then service the
            # newest requested electrode. This avoids competing disk scans.
            self._update_range_fast_access_ui()
            return

        self._fast_electrode_name = electrode
        self._fast_electrode_progress = 0
        thread = QThread(self)
        worker = FastElectrodeCacheWorker(self.state.recording, electrode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_fast_electrode_progress)
        worker.finished.connect(self._on_fast_electrode_ready)
        worker.failed.connect(self._on_fast_electrode_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._fast_electrode_thread_finished)
        self._fast_electrode_thread = thread
        self._fast_electrode_worker = worker
        thread.start()
        self._update_range_fast_access_ui()

    def _on_fast_electrode_progress(self, electrode: str, pct: int):
        if electrode == self._fast_electrode_name:
            self._fast_electrode_progress = int(pct)
            self._update_range_fast_access_ui()

    def _on_fast_electrode_ready(self, electrode: str):
        self._fast_electrode_failed.discard(electrode)
        if electrode == self._fast_electrode_name:
            self._fast_electrode_progress = 100
        self._refresh_recording_details()
        if self.selected_preview_electrode == electrode:
            self._show_large_preview(electrode)
        self._update_range_fast_access_ui()
        QTimer.singleShot(150, self._start_background_preload)

    def _on_fast_electrode_failed(self, electrode: str, details: str):
        self._fast_electrode_failed.add(electrode)
        self._update_range_fast_access_ui()
        if self.state.recording.source_format.casefold() == "csv" and self.state.recording.cache_parquet is None:
            QMessageBox.critical(
                self,
                "CSV electrode preparation error",
                "The selected electrode could not be prepared from the CSV.\n\n" + details,
            )

    def _fast_electrode_thread_finished(self):
        finished = self._fast_electrode_name
        self._fast_electrode_thread = None
        self._fast_electrode_worker = None
        self._fast_electrode_name = None
        self._fast_electrode_progress = 0
        requested = self._fast_electrode_requested
        self._update_range_fast_access_ui()
        if requested and requested != finished and requested == self.state.selected_electrode:
            QTimer.singleShot(0, lambda e=requested: self._start_fast_electrode_preparation(e))

    def _update_range_fast_access_ui(self):
        if not hasattr(self, "range_apply_btn"):
            return
        electrode = self.state.selected_electrode
        recording = self.state.recording
        if not electrode or recording.source_path is None:
            self.range_apply_btn.setText("Use range → Detection")
            self.range_apply_btn.setEnabled(False)
            return

        lazy_csv = (
            recording.source_format.casefold() == "csv"
            and recording.cache_parquet is None
        )

        if self._range_load_thread is not None:
            pct = max(0, min(99, int(self._range_load_progress)))
            self.range_apply_btn.setText(f"Loading selected range… {pct}%")
            self.range_apply_btn.setEnabled(False)
            self.range_apply_btn.setToolTip(
                "Reading the chosen interval at full resolution for scientific analysis."
            )
            return

        if lazy_csv:
            # A completed older sidecar is still a valid fast path, but Build 20
            # does not create one. Normally we only wait for the lightweight 30k
            # Range index, then Apply reads the selected interval at full resolution.
            if fast_electrode_cache_ready(recording, electrode):
                self.range_apply_btn.setText("Use range → Detection")
                self.range_apply_btn.setEnabled(True)
                self.range_apply_btn.setToolTip("Existing full-resolution electrode cache is ready.")
                return

            if load_range_preview(recording, electrode) is not None:
                self.range_apply_btn.setText("Use range → Detection")
                self.range_apply_btn.setEnabled(True)
                self.range_apply_btn.setToolTip(
                    "Only the selected interval will now be read at full resolution."
                )
                return

            if electrode in self._range_preview_failed:
                self.range_apply_btn.setText("Use range → Detection")
                self.range_apply_btn.setEnabled(True)
                self.range_apply_btn.setToolTip(
                    "Enhanced preview was unavailable; full-resolution loading may need to scan farther through the CSV."
                )
                return

            if self._range_preview_thread is not None and self._range_preview_electrode == electrode:
                pct = max(0, min(99, int(self._range_preview_progress)))
                self.range_apply_btn.setText(f"Preparing range preview… {pct}%")
                self.range_apply_btn.setEnabled(False)
                self.range_apply_btn.setToolTip(
                    "You can set the range now. The blue continue button becomes available as soon as the ~30k-point preview/index is ready."
                )
                return

            self.range_apply_btn.setText("Preparing range preview…")
            self.range_apply_btn.setEnabled(False)
            QTimer.singleShot(0, lambda e=electrode: self._start_range_preview_preparation(e))
            return

        # XLSX / Parquet path retained from the approved Build 14/15 behavior.
        if fast_electrode_cache_ready(recording, electrode):
            self.range_apply_btn.setText("Use range → Detection")
            self.range_apply_btn.setEnabled(True)
            self.range_apply_btn.setToolTip("Fast random-access electrode cache is ready.")
            return
        if electrode in self._fast_electrode_failed:
            self.range_apply_btn.setText("Use range → Detection")
            self.range_apply_btn.setEnabled(recording.cache_parquet is not None)
            self.range_apply_btn.setToolTip(
                "Fast sidecar unavailable; using the standard Parquet range reader."
            )
            return
        if self._fast_electrode_thread is not None and self._fast_electrode_name == electrode:
            pct = max(0, min(99, int(self._fast_electrode_progress)))
            self.range_apply_btn.setText(f"Preparing fast access… {pct}%")
            self.range_apply_btn.setEnabled(False)
            self.range_apply_btn.setToolTip(
                "You can choose/adjust the range while the selected electrode is optimized in the background."
            )
            return
        self.range_apply_btn.setText("Preparing fast access…")
        self.range_apply_btn.setEnabled(False)
        QTimer.singleShot(0, lambda e=electrode: self._start_fast_electrode_preparation(e))

    def choose_output_dir(self):
        start = self.output_dir_edit.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Choose results folder", start)
        if directory:
            self.output_dir_edit.setText(directory)

    def _output_changed(self):
        self.state.analysis_name = self.analysis_name_edit.text().strip()
        path = self.output_dir_edit.text().strip()
        self.state.output_dir = Path(path) if path else None
        self.refresh_output_preview()
        self.refresh_all()

    def refresh_output_preview(self):
        name = self.analysis_name_edit.text().strip() or "analysis"
        directory = self.output_dir_edit.text().strip() or "—"
        self.output_preview.setText(
            f"Final files:\n{directory}/{name}.xlsx\n{directory}/{name}.pdf"
        )

    def refresh_cache_page(self):
        entries = self.cache_manager.entries()
        self.cache_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            name_item = QTableWidgetItem(entry.source_name)
            name_item.setData(Qt.UserRole, str(entry.directory))
            name_item.setToolTip(entry.source_path)
            self.cache_table.setItem(row, 0, name_item)
            self.cache_table.setItem(row, 1, QTableWidgetItem(entry.last_used_text))
            size_item = QTableWidgetItem(human_size(entry.size_bytes))
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.cache_table.setItem(row, 2, size_item)
        total = sum(e.size_bytes for e in entries)
        self.cache_total_label.setText(f"Total cache: {human_size(total)}")
        self._refresh_cache_nav(total)

    def clear_selected_cache(self):
        rows = sorted({idx.row() for idx in self.cache_table.selectionModel().selectedRows()})
        if not rows:
            return
        dirs = []
        for row in rows:
            item = self.cache_table.item(row, 0)
            if item:
                dirs.append(Path(item.data(Qt.UserRole)))
        self.cache_manager.clear_entries(dirs)
        self.refresh_cache_page()

    def clear_all_cache(self):
        if not self.cache_manager.entries():
            return
        answer = QMessageBox.question(
            self,
            "Clear cache",
            "Clear all prepared recordings? Original CSV/XLSX files and analysis results will not be changed.",
        )
        if answer == QMessageBox.Yes:
            self.cache_manager.clear_all()
            self.refresh_cache_page()

    def _refresh_cache_nav(self, total_bytes):
        button = self.nav_buttons.get("cache")
        if not button:
            return
        if total_bytes > CACHE_WARNING_BYTES:
            button.setText(f"Cache  •  {human_size(total_bytes)}")
            button.setStyleSheet(
                "QPushButton {text-align:left; padding:8px 12px; border:none; border-radius:7px;"
                " font-size:13px; color:#b91c1c; background:#fee2e2; font-weight:800;}"
                "QPushButton:hover {background:#fecaca;}"
                "QPushButton:checked {background:#fecaca; color:#991b1b;}"
            )
        else:
            button.setText("Cache")
            button.setStyleSheet(
                "QPushButton {text-align:left; padding:8px 12px; border:none; border-radius:7px;"
                " font-size:13px; color:#1f2937;}"
                "QPushButton:hover {background:#eef2ff;}"
                "QPushButton:checked {background:#dbeafe; color:#1d4ed8; font-weight:700;}"
            )

    def clear_selected_cache_if_current(self):
        pass

    def refresh_all(self):
        rec = self.state.recording
        file_name = rec.source_path.name if rec.source_path else "—"
        display_file = file_name if len(file_name) <= 18 else file_name[:15] + "…"
        self.summary_file.setText(f"File\n{display_file}")
        self.summary_file.setToolTip(str(rec.source_path) if rec.source_path else "")
        self.summary_electrode.setText(f"Electrode\n{self.state.selected_electrode or '—'}")
        if self.state.selected_range:
            range_text = f"{self.state.selected_range[0]:g}–{self.state.selected_range[1]:g} ms"
        else:
            range_text = "—"
        self.summary_range.setText(f"Range\n{range_text}")
        threshold_text = "—" if self.state.threshold is None else f"{self.state.threshold:g}"
        self.summary_threshold.setText(f"Threshold\n{threshold_text}")

        self._set_setup_nav_status("recording", bool(rec.source_path))
        electrode_complete = bool(self.state.selected_electrode)
        self._set_setup_nav_status("electrode", electrode_complete)
        self._refresh_analysis_nav()
        self.refresh_output_preview()
        self._update_restart_controls()

    def _set_setup_nav_status(self, key, complete):
        labels = {"recording": "Recording", "electrode": "Electrode"}
        icon = "✓" if complete else "○"
        self.nav_buttons[key].setText(f"{icon}  {labels[key]}")

    def _refresh_analysis_nav(self):
        for i, step in enumerate(ANALYSIS_STEPS, start=1):
            state = self.state.step_states[step]
            if state == StepState.COMPLETE:
                prefix = "✓"
                suffix = ""
            elif state == StepState.NEEDS_UPDATE:
                prefix = "↻"
                suffix = "  Needs update"
            elif state == StepState.SKIPPED:
                prefix = "–"
                suffix = ""
            else:
                prefix = "○"
                suffix = ""
            self.nav_buttons[step].setText(f"{prefix}  {i}  {step.title()}{suffix}")


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("MEA Analyzer")
    app.setOrganizationName("MEA Analyzer")
    app.setStyle("Fusion")
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
