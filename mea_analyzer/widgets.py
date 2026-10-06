from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)


class SparklineWidget(QWidget):
    def __init__(self, parent=None, max_points: int | None = None):
        super().__init__(parent)
        self._x = np.array([], dtype=float)
        self._y = np.array([], dtype=float)
        self._max_points = max_points
        self.setMinimumHeight(32)

    def set_data(self, x, y):
        x_arr = np.asarray(x if x is not None else [], dtype=float)
        y_arr = np.asarray(y if y is not None else [], dtype=float)
        n = min(x_arr.size, y_arr.size)
        x_arr, y_arr = x_arr[:n], y_arr[:n]
        if self._max_points and n > self._max_points:
            # Mini-plots are navigation previews only. Drawing hundreds of
            # thousands of path segments across a 64-electrode grid wastes GUI
            # time, so retain an evenly sampled visual preview for painting.
            idx = np.linspace(0, n - 1, self._max_points, dtype=int)
            x_arr = x_arr[idx]
            y_arr = y_arr[idx]
        self._x = x_arr
        self._y = y_arr
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        if len(self._y) < 2:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(rect, Qt.AlignCenter, "No preview")
            return
        finite = np.isfinite(self._x) & np.isfinite(self._y)
        if finite.sum() < 2:
            return
        x = self._x[finite]
        y = self._y[finite]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if math.isclose(x_min, x_max):
            x_max = x_min + 1.0
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0
        path = QPainterPath()
        for i, (xx, yy) in enumerate(zip(x, y)):
            px = rect.left() + (xx - x_min) / (x_max - x_min) * rect.width()
            py = rect.bottom() - (yy - y_min) / (y_max - y_min) * rect.height()
            if i == 0:
                path.moveTo(QPointF(px, py))
            else:
                path.lineTo(QPointF(px, py))
        painter.setPen(QPen(QColor("#111827"), 1.1))
        painter.drawPath(path)


class ElectrodeCard(QFrame):
    clicked = Signal(str)

    def __init__(self, electrode_name: str, x, y, parent=None, usable: bool = True):
        super().__init__(parent)
        self.electrode_name = electrode_name
        self._selected = False
        self._usable = bool(usable)
        self.setCursor(Qt.PointingHandCursor if self._usable else Qt.ArrowCursor)
        self.setFixedSize(96, 78)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(1)
        label = QLabel(electrode_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 10px; font-weight: 600;")
        self.spark = SparklineWidget(max_points=650)
        self.spark.set_data(x, y)
        layout.addWidget(label)
        layout.addWidget(self.spark, 1)
        if not self._usable:
            no_data = QLabel("No data")
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setStyleSheet("font-size:9px; color:#9ca3af;")
            layout.addWidget(no_data)
        self._refresh_style()

    def set_selected(self, selected: bool):
        self._selected = selected
        self._refresh_style()

    def _refresh_style(self):
        if not self._usable:
            self.setStyleSheet(
                "QFrame {border: 1px solid #e5e7eb; border-radius: 7px; background: #f9fafb;}"
            )
        elif self._selected:
            self.setStyleSheet(
                "QFrame {border: 3px solid #2563eb; border-radius: 7px; background: #eff6ff;}"
            )
        else:
            self.setStyleSheet(
                "QFrame {border: 1px solid #d1d5db; border-radius: 7px; background: white;}"
            )

    def mousePressEvent(self, event):
        if self._usable and event.button() == Qt.LeftButton:
            self.clicked.emit(self.electrode_name)
        super().mousePressEvent(event)


class LargeSignalPreview(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("largePreview")
        self.setStyleSheet(
            "#largePreview {border: 1px solid #d1d5db; border-radius: 10px; background: white;}"
        )
        layout = QVBoxLayout(self)
        self.title = QLabel("Select an electrode")
        self.title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.spark = SparklineWidget(max_points=5000)
        self.spark.setMinimumHeight(220)
        self.meta = QLabel("")
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet("color: #4b5563;")
        layout.addWidget(self.title)
        layout.addWidget(self.spark, 1)
        layout.addWidget(self.meta)

    def set_signal(self, name: str, x, y, row_count: int, row_count_exact: bool = True):
        self.title.setText(name)
        self.spark.set_data(x, y)
        finite = np.asarray(y, dtype=float)
        finite = finite[np.isfinite(finite)]
        count_label = f"{row_count:,}" if row_count_exact else f"~{row_count:,} (estimated)"
        if finite.size:
            self.meta.setText(
                f"Samples in recording: {count_label}\n"
                f"Preview range: {float(np.min(finite)):.4g} to {float(np.max(finite)):.4g}"
            )
        else:
            self.meta.setText(f"Samples in recording: {count_label}")

# -----------------------------------------------------------------------------
# Range-selection graph used by Analysis Step 1
# -----------------------------------------------------------------------------
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RangeSelectionPlot(QFrame):
    """Embedded signal viewer with draggable start/end boundaries.

    Only a downsampled representation is drawn. The full-resolution arrays remain
    in memory and are used when the selected range is applied.
    """

    selectionChanged = Signal(float, float)
    MAX_DISPLAY_POINTS = 30_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame {border:1px solid #d1d5db; border-radius:10px; background:white;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(figsize=(8, 4), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self._time = np.array([], dtype=float)
        self._voltage = np.array([], dtype=float)
        self._start = None
        self._end = None
        self._full_min = None
        self._full_max = None
        self._start_line = None
        self._end_line = None
        self._span = None
        self._dragging = None

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    @property
    def selection(self):
        if self._start is None or self._end is None:
            return None
        return float(self._start), float(self._end)

    def set_signal(self, time_values, voltage_values, start=None, end=None, full_start=None, full_end=None):
        self._time = np.asarray(time_values if time_values is not None else [], dtype=float)
        self._voltage = np.asarray(voltage_values if voltage_values is not None else [], dtype=float)
        # ax.clear() invalidates the previous Matplotlib artists. Reset our
        # references as well so revisiting Range or starting the next file never
        # tries to manipulate an artist belonging to the previous analysis.
        self._start_line = None
        self._end_line = None
        self._span = None
        self._dragging = None
        self.ax.clear()

        finite = np.isfinite(self._time) & np.isfinite(self._voltage)
        if finite.sum() < 2:
            self.ax.text(0.5, 0.5, "No usable signal data", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        t = self._time[finite]
        v = self._voltage[finite]
        stride = max(1, int(math.ceil(len(t) / self.MAX_DISPLAY_POINTS)))
        td = t[::stride]
        vd = v[::stride]

        self.ax.plot(td, vd, linewidth=0.7, color="#111827")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Potential (mV)")
        self.ax.set_title("Select the data range for analysis")
        self.ax.grid(False)

        preview_min = float(np.min(t))
        preview_max = float(np.max(t))
        t_min = preview_min if full_start is None else float(full_start)
        t_max = preview_max if full_end is None else float(full_end)
        if t_max < t_min:
            t_min, t_max = t_max, t_min
        self._full_min, self._full_max = t_min, t_max
        start = t_min if start is None else float(np.clip(start, t_min, t_max))
        end = t_max if end is None else float(np.clip(end, t_min, t_max))
        if start > end:
            start, end = end, start
        self._start, self._end = start, end

        self._start_line = self.ax.axvline(start, color="#dc2626", linestyle="--", linewidth=1.8, label="Start")
        self._end_line = self.ax.axvline(end, color="#2563eb", linestyle="--", linewidth=1.8, label="End")
        self._draw_span()
        self.ax.legend(loc="upper right")
        self.ax.set_xlim(t_min, t_max)
        self.canvas.draw_idle()

    def clear_signal(self):
        """Return the reusable Range widget to a clean pre-recording state."""
        self._time = np.array([], dtype=float)
        self._voltage = np.array([], dtype=float)
        self._start = None
        self._end = None
        self._full_min = None
        self._full_max = None
        self._start_line = None
        self._end_line = None
        self._span = None
        self._dragging = None
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Select a recording and electrode", ha="center", va="center", transform=self.ax.transAxes, color="#6b7280")
        self.canvas.draw_idle()

    def set_selection(self, start, end, emit=False):
        if self._time.size < 2:
            return
        finite_t = self._time[np.isfinite(self._time)]
        if finite_t.size < 2:
            return
        t_min = self._full_min if self._full_min is not None else float(np.min(finite_t))
        t_max = self._full_max if self._full_max is not None else float(np.max(finite_t))
        start = float(np.clip(start, t_min, t_max))
        end = float(np.clip(end, t_min, t_max))
        if start > end:
            start, end = end, start
        self._start, self._end = start, end
        if self._start_line is not None:
            self._start_line.set_xdata([start, start])
        if self._end_line is not None:
            self._end_line.set_xdata([end, end])
        self._draw_span()
        self.canvas.draw_idle()
        if emit:
            self.selectionChanged.emit(start, end)

    def reset_view(self):
        if self._time.size:
            finite = self._time[np.isfinite(self._time)]
            if finite.size:
                left = self._full_min if self._full_min is not None else float(np.min(finite))
                right = self._full_max if self._full_max is not None else float(np.max(finite))
                self.ax.set_xlim(left, right)
                self.canvas.draw_idle()

    def _draw_span(self):
        if self._span is not None:
            try:
                self._span.remove()
            except ValueError:
                pass
        if self._start is not None and self._end is not None:
            self._span = self.ax.axvspan(self._start, self._end, color="#93c5fd", alpha=0.16, zorder=0)

    def _nearest_boundary(self, event):
        """Return the closest boundary for any click inside the plot.

        Build 20 deliberately removes the small hit target: the user can click
        anywhere and the nearest boundary is the one that moves.
        """
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or self._start is None
            or self._end is None
        ):
            return None
        yref = 0.0
        start_px = self.ax.transData.transform((self._start, yref))[0]
        end_px = self.ax.transData.transform((self._end, yref))[0]
        ds = abs(event.x - start_px)
        de = abs(event.x - end_px)
        return "start" if ds <= de else "end"

    def _move_boundary(self, boundary, x, emit=True):
        if self._start is None or self._end is None:
            return
        full_width = max(
            1.0,
            float(self._full_max - self._full_min)
            if self._full_min is not None and self._full_max is not None
            else abs(float(self._end - self._start)),
        )
        epsilon = max(np.finfo(float).eps * full_width * 16.0, full_width * 1e-12)
        if boundary == "start":
            new_start = min(float(x), float(self._end) - epsilon)
            self.set_selection(new_start, self._end, emit=emit)
        elif boundary == "end":
            new_end = max(float(x), float(self._start) + epsilon)
            self.set_selection(self._start, new_end, emit=emit)

    def _on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax or event.xdata is None:
            return
        self._dragging = self._nearest_boundary(event)
        if self._dragging:
            # A normal click is sufficient: move the nearest boundary immediately.
            # Keeping _dragging set also makes click-and-drag work naturally.
            self._move_boundary(self._dragging, float(event.xdata), emit=True)

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        self._move_boundary(self._dragging, float(event.xdata), emit=True)

    def _on_release(self, event):
        self._dragging = None

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        left, right = self.ax.get_xlim()
        if right <= left:
            return
        factor = 0.8 if event.button == "up" else 1.25
        cursor = float(event.xdata)
        new_left = cursor - (cursor - left) * factor
        new_right = cursor + (right - cursor) * factor

        finite = self._time[np.isfinite(self._time)]
        if finite.size:
            full_left = self._full_min if self._full_min is not None else float(np.min(finite))
            full_right = self._full_max if self._full_max is not None else float(np.max(finite))
            if new_right - new_left >= full_right - full_left:
                new_left, new_right = full_left, full_right
            else:
                width = new_right - new_left
                if new_left < full_left:
                    new_left, new_right = full_left, full_left + width
                if new_right > full_right:
                    new_right, new_left = full_right, full_right - width
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()


class DetectionPlot(QFrame):
    """Embedded v7 smoothing + Z-score threshold viewer.

    Full-resolution arrays are used for calculations. Only a decimated copy is
    rendered so interaction stays responsive on long selected ranges.
    """

    thresholdChanged = Signal(float)
    MAX_DISPLAY_POINTS = 30_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame {border:1px solid #d1d5db; border-radius:10px; background:white;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(figsize=(8, 5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.signal_ax, self.z_ax = self.figure.subplots(2, 1, sharex=True)
        layout.addWidget(self.canvas, 1)

        self._time = np.array([], dtype=float)
        self._raw = np.array([], dtype=float)
        self._filtered = np.array([], dtype=float)
        self._z = np.array([], dtype=float)
        self._threshold = 3.0
        self._threshold_line = None
        self._spike_artist = None
        self._dragging_threshold = False

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    @property
    def threshold(self):
        return float(self._threshold)

    def set_data(
        self,
        time_values,
        raw_voltage,
        filtered_voltage,
        z_scores,
        threshold,
        spike_indices=None,
        smoothing_enabled=True,
    ):
        self._time = np.asarray(time_values if time_values is not None else [], dtype=float)
        self._raw = np.asarray(raw_voltage if raw_voltage is not None else [], dtype=float)
        self._filtered = np.asarray(filtered_voltage if filtered_voltage is not None else [], dtype=float)
        self._z = np.asarray(z_scores if z_scores is not None else [], dtype=float)
        self._threshold = float(threshold)

        self.signal_ax.clear()
        self.z_ax.clear()
        if self._time.size < 2 or self._z.size != self._time.size:
            self.z_ax.text(
                0.5, 0.5, "No detection data", ha="center", va="center", transform=self.z_ax.transAxes
            )
            self.canvas.draw_idle()
            return

        finite = (
            np.isfinite(self._time)
            & np.isfinite(self._raw)
            & np.isfinite(self._filtered)
            & np.isfinite(self._z)
        )
        if finite.sum() < 2:
            self.z_ax.text(
                0.5, 0.5, "No usable detection data", ha="center", va="center", transform=self.z_ax.transAxes
            )
            self.canvas.draw_idle()
            return

        indices = np.flatnonzero(finite)
        stride = max(1, int(math.ceil(len(indices) / self.MAX_DISPLAY_POINTS)))
        display_idx = indices[::stride]

        self.signal_ax.plot(
            self._time[display_idx], self._raw[display_idx],
            linewidth=0.6, alpha=0.45, color="#6b7280", label="Raw"
        )
        if smoothing_enabled:
            self.signal_ax.plot(
                self._time[display_idx], self._filtered[display_idx],
                linewidth=0.9, color="#111827", label="Filtered"
            )
        self.signal_ax.set_ylabel("Potential (mV)")
        self.signal_ax.set_title("Signal used for detection")
        self.signal_ax.legend(loc="upper right", fontsize=8)

        self.z_ax.plot(
            self._time[display_idx], self._z[display_idx],
            linewidth=0.7, color="#2563eb", label="Z-score"
        )
        self._threshold_line = self.z_ax.axhline(
            self._threshold, color="#dc2626", linestyle="--", linewidth=1.5,
            label=f"Threshold = {self._threshold:.2f}"
        )
        self._spike_artist = None
        self.set_spikes(spike_indices, redraw=False)
        self.z_ax.set_xlabel("Time (ms)")
        self.z_ax.set_ylabel("Z-score")
        self.z_ax.set_title("Click or drag in this panel to set the threshold")
        self.z_ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def set_threshold(self, threshold, redraw=True):
        self._threshold = float(threshold)
        if self._threshold_line is not None:
            self._threshold_line.set_ydata([self._threshold, self._threshold])
            self._threshold_line.set_label(f"Threshold = {self._threshold:.2f}")
            self.z_ax.legend(loc="upper right", fontsize=8)
        if redraw:
            self.canvas.draw_idle()

    def set_spikes(self, spike_indices, redraw=True):
        if self._spike_artist is not None:
            try:
                self._spike_artist.remove()
            except ValueError:
                pass
            self._spike_artist = None

        if spike_indices is not None and self._time.size and self._z.size:
            idx = np.asarray(spike_indices, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < self._time.size)]
            if idx.size:
                self._spike_artist = self.z_ax.scatter(
                    self._time[idx], self._z[idx], s=14, color="#dc2626", zorder=4, label="Crossings"
                )
                self.z_ax.legend(loc="upper right", fontsize=8)
        if redraw:
            self.canvas.draw_idle()

    def _set_threshold_from_event(self, event):
        if event.inaxes != self.z_ax or event.ydata is None:
            return
        value = float(event.ydata)
        self.set_threshold(value, redraw=True)
        self.thresholdChanged.emit(value)

    def _on_press(self, event):
        if event.button == 1 and event.inaxes == self.z_ax:
            self._dragging_threshold = True
            self._set_threshold_from_event(event)

    def _on_motion(self, event):
        if self._dragging_threshold:
            self._set_threshold_from_event(event)

    def _on_release(self, event):
        self._dragging_threshold = False


class SnippetAveragingPlot(QFrame):
    """Embedded replacement for v7 InteractiveSnippetOverlayPlot.

    Scientific behavior is unchanged: all snippets begin included, the mean is
    computed across included snippets, and users can toggle individual snippets.
    The old plot legend is replaced by a clear vertical checkbox legend so every
    beat can be included/excluded without hunting for a line in the graph.
    """

    selectionChanged = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame {border:1px solid #d1d5db; border-radius:10px; background:white;}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.figure = Figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        layout.addWidget(self.canvas, 1)

        self.beat_panel = QFrame()
        self.beat_panel.setFixedWidth(132)
        self.beat_panel.setStyleSheet(
            "QFrame {background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;}"
        )
        panel_layout = QVBoxLayout(self.beat_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(5)
        title = QLabel("BEATS")
        title.setStyleSheet("border:none; font-size:10px; font-weight:800; color:#6b7280;")
        panel_layout.addWidget(title)

        self.beat_scroll = QScrollArea()
        self.beat_scroll.setWidgetResizable(True)
        self.beat_scroll.setFrameShape(QFrame.NoFrame)
        self.beat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.beat_list_widget = QWidget()
        self.beat_list_layout = QVBoxLayout(self.beat_list_widget)
        self.beat_list_layout.setContentsMargins(0, 0, 0, 0)
        self.beat_list_layout.setSpacing(3)
        self.beat_list_layout.addStretch(1)
        self.beat_scroll.setWidget(self.beat_list_widget)
        panel_layout.addWidget(self.beat_scroll, 1)

        self.mean_label = QLabel("Mean")
        self.mean_label.setStyleSheet("border:none; color:#111827; font-weight:800; font-size:11px;")
        panel_layout.addWidget(self.mean_label)
        layout.addWidget(self.beat_panel, 0)

        self._times = np.array([], dtype=float)
        self._snippets = []
        self._selected = np.array([], dtype=bool)
        self._snippet_lines = []
        self._line_map = {}
        self._checkboxes = []
        self._mean_line = None
        self._block_idx = None
        self._snippet_length_ms = 0.0
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def set_block(self, block_idx, snippet_times, snippets, selected_mask, mean_snippet, snippet_length_ms):
        self._block_idx = int(block_idx)
        self._times = np.asarray(snippet_times if snippet_times is not None else [], dtype=float)
        self._snippets = [np.asarray(s, dtype=float) for s in (snippets or [])]
        self._selected = np.asarray(selected_mask, dtype=bool).copy()
        self._snippet_length_ms = float(snippet_length_ms or 0.0)
        self._draw(np.asarray(mean_snippet if mean_snippet is not None else [], dtype=float))

    def _clear_checkboxes(self):
        while self.beat_list_layout.count() > 1:
            item = self.beat_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes = []

    def _draw(self, mean_snippet):
        from matplotlib.colors import to_hex

        self.ax.clear()
        self._snippet_lines = []
        self._line_map = {}
        self._mean_line = None
        self._clear_checkboxes()

        if self._times.size == 0 or not self._snippets:
            self.ax.text(0.5, 0.5, "No valid snippets in this group", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        for i, snippet in enumerate(self._snippets):
            selected = bool(i < len(self._selected) and self._selected[i])
            line, = self.ax.plot(
                self._times,
                snippet,
                linewidth=0.9,
                alpha=0.62 if selected else 0.10,
                picker=5,
            )
            self._snippet_lines.append(line)
            self._line_map[line] = i

            color = to_hex(line.get_color())
            checkbox = QCheckBox(f"Beat {i + 1}")
            checkbox.setChecked(selected)
            checkbox.setCursor(Qt.PointingHandCursor)
            checkbox.setStyleSheet(
                f"QCheckBox {{border:none; color:{color}; font-size:11px; font-weight:700;}}"
                "QCheckBox::indicator {width:14px; height:14px;}"
            )
            checkbox.toggled.connect(lambda checked, idx=i: self.selectionChanged.emit(idx, checked))
            self.beat_list_layout.insertWidget(self.beat_list_layout.count() - 1, checkbox)
            self._checkboxes.append(checkbox)

        if mean_snippet.size == self._times.size:
            self._mean_line, = self.ax.plot(
                self._times,
                mean_snippet,
                color="black",
                linewidth=2.0,
            )

        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.set_title(f"Beat overlay — Group {self._block_idx}")
        self.ax.text(
            0.99,
            0.02,
            f"Snippet length: {self._snippet_length_ms:.2f} ms",
            ha="right",
            va="bottom",
            transform=self.ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.12),
        )
        self.canvas.draw_idle()

    def update_mean(self, selected_mask, mean_snippet):
        self._selected = np.asarray(selected_mask, dtype=bool).copy()
        for i, line in enumerate(self._snippet_lines):
            line.set_alpha(0.62 if i < len(self._selected) and self._selected[i] else 0.10)
        for i, checkbox in enumerate(self._checkboxes):
            checked = bool(i < len(self._selected) and self._selected[i])
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        mean = np.asarray(mean_snippet, dtype=float)
        if self._mean_line is not None and mean.size == self._times.size:
            self._mean_line.set_ydata(mean)
        self.canvas.draw_idle()

    def _on_pick(self, event):
        idx = self._line_map.get(event.artist)
        if idx is None or idx >= len(self._selected):
            return
        new_value = not bool(self._selected[idx])
        self.selectionChanged.emit(int(idx), bool(new_value))


class BaselinePlot(QFrame):
    """Interactive in-app version of the v7 baseline selection plot."""

    baselineChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame {border:1px solid #d1d5db; border-radius:10px; background:white;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.figure = Figure(figsize=(8, 4.5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        layout.addWidget(self.canvas, 1)

        self._time = np.array([], dtype=float)
        self._voltage = np.array([], dtype=float)
        self._baseline = None
        self._final_line = None
        self._block_idx = None
        self.canvas.mpl_connect("button_press_event", self._on_click)

    @property
    def baseline(self):
        return None if self._baseline is None else float(self._baseline)

    def set_group(self, block_idx, time_values, voltage_values, baseline_pre, baseline_post, selected_baseline):
        self._block_idx = int(block_idx)
        self._time = np.asarray(time_values if time_values is not None else [], dtype=float)
        self._voltage = np.asarray(voltage_values if voltage_values is not None else [], dtype=float)
        self._baseline = None if selected_baseline is None else float(selected_baseline)
        self.ax.clear()

        if self._time.size == 0 or self._voltage.size == 0:
            self.ax.text(0.5, 0.5, "No mean waveform available", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        self.ax.plot(self._time, self._voltage, color="gray", alpha=0.75, linewidth=1.0, label="Mean waveform")
        self.ax.axhline(float(baseline_pre), color="#2563eb", linestyle="--", linewidth=1.4, label="Pre-spike baseline")
        self.ax.axhline(float(baseline_post), color="#dc2626", linestyle="--", linewidth=1.4, label="Post-spike baseline")
        self._final_line = self.ax.axhline(
            float(self._baseline), color="#111827", linestyle="-", linewidth=2.0, label="Final baseline"
        )
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.set_title(f"Baseline selection — Group {self._block_idx}")
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def set_baseline(self, value, emit=False):
        if value is None:
            return
        self._baseline = float(value)
        if self._final_line is not None:
            self._final_line.set_ydata([self._baseline, self._baseline])
            self.canvas.draw_idle()
        if emit:
            self.baselineChanged.emit(self._baseline)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.ydata is None:
            return
        self.set_baseline(float(event.ydata), emit=True)



class LandmarksPlot(QFrame):
    """Interactive mean-waveform plot for the v7 landmark adjustment step."""

    markerMoved = Signal(str, float)

    COLORS = {
        "spike_max": "#dc2626",
        "spike_min": "#991b1b",
        "spike_start": "#7e22ce",
        "t_wave_start": "#2563eb",
        "t_wave_peak": "#1e3a8a",
        "t_wave_end": "#0284c7",
        "EAD": "#111827",
    }

    LABELS = {
        "spike_max": "Spike max",
        "spike_min": "Spike min",
        "spike_start": "Spike start",
        "t_wave_start": "T-wave start",
        "t_wave_peak": "T-wave peak",
        "t_wave_end": "T-wave end",
        "EAD": "EAD",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame {border:1px solid #d1d5db; border-radius:10px; background:white;}"
        )
        layout = QVBoxLayout(self)
        # Keep the frame almost flush with the canvas so horizontal space is
        # spent on the waveform rather than decorative side padding.
        layout.setContentsMargins(2, 2, 2, 2)
        self.figure = Figure(figsize=(8, 5.5), dpi=100, tight_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.subplots_adjust(left=0.085, right=0.995, bottom=0.13, top=0.92)
        layout.addWidget(self.canvas, 1)

        # Horizontal navigation stays lightweight and only changes the visible
        # x-window. It never alters landmark values or the underlying waveform.
        self.hscroll = QScrollBar(Qt.Horizontal)
        self.hscroll.setRange(0, 10_000)
        self.hscroll.setPageStep(10_000)
        self.hscroll.setSingleStep(150)
        self.hscroll.setEnabled(False)
        self.hscroll.setToolTip("Drag to move left/right through the waveform when zoomed.")
        layout.addWidget(self.hscroll)
        self.hscroll.valueChanged.connect(self._on_scrollbar_changed)
        self._updating_scrollbar = False

        self._time = np.array([], dtype=float)
        self._voltage = np.array([], dtype=float)
        self._values = {}
        self._baseline = None
        self._flags = {"exclude_spectrum": False}
        self._enabled = {
            "spike_max": True, "spike_min": True, "spike_start": True,
            "t_wave_start": False, "t_wave_peak": False, "t_wave_end": False, "EAD": False,
        }
        self._selected_marker = "spike_max"
        self._marker_lines = {}
        self._waveform_line = None
        self._block_idx = None
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def set_group(self, block_idx, time_values, voltage_values, baseline, values, flags, enabled=None, selected_marker=None):
        self._block_idx = int(block_idx)
        self._time = np.asarray(time_values if time_values is not None else [], dtype=float)
        self._voltage = np.asarray(voltage_values if voltage_values is not None else [], dtype=float)
        self._baseline = float(baseline)
        self._values = {k: float(v) for k, v in (values or {}).items()}
        self._flags = {"exclude_spectrum": bool((flags or {}).get("exclude_spectrum", False))}
        defaults = {
            "spike_max": True, "spike_min": True, "spike_start": True,
            "t_wave_start": False, "t_wave_peak": False, "t_wave_end": False, "EAD": False,
        }
        self._enabled = {key: bool((enabled or defaults).get(key, defaults[key])) for key in self.COLORS}
        if selected_marker in self.COLORS:
            self._selected_marker = selected_marker
        self._draw()

    def set_selected_marker(self, marker_key):
        if marker_key not in self.COLORS:
            return
        self._selected_marker = marker_key
        self._refresh_marker_styles()
        self.canvas.draw_idle()

    def set_marker_value(self, marker_key, value, redraw=True):
        if marker_key not in self.COLORS or not np.isfinite(value):
            return
        self._values[marker_key] = float(value)
        line = self._marker_lines.get(marker_key)
        if line is not None:
            line.set_xdata([float(value), float(value)])
        if redraw:
            self.canvas.draw_idle()

    def set_flags(self, flags):
        self._flags = {"exclude_spectrum": bool((flags or {}).get("exclude_spectrum", False))}
        self._refresh_visibility()
        self.canvas.draw_idle()

    def set_enabled(self, enabled):
        defaults = {
            "spike_max": True, "spike_min": True, "spike_start": True,
            "t_wave_start": False, "t_wave_peak": False, "t_wave_end": False, "EAD": False,
        }
        self._enabled = {key: bool((enabled or defaults).get(key, defaults[key])) for key in self.COLORS}
        self._refresh_visibility()
        self.canvas.draw_idle()

    def _draw(self):
        self.ax.clear()
        self._marker_lines = {}
        self._waveform_line = None
        if self._time.size == 0 or self._voltage.size == 0:
            self.ax.text(0.5, 0.5, "No mean waveform available", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        self._waveform_line, = self.ax.plot(
            self._time, self._voltage, color="#111827", alpha=0.82, linewidth=1.2, label="Mean waveform"
        )
        self.ax.axhline(self._baseline, color="#6b7280", linewidth=1.2, alpha=0.7, label="Baseline")

        for key in ["spike_max", "spike_min", "spike_start", "t_wave_start", "t_wave_peak", "t_wave_end", "EAD"]:
            if key not in self._values:
                continue
            linestyle = "-" if key == "EAD" else "--"
            linewidth = 2.5 if key == "EAD" else 1.8
            self._marker_lines[key] = self.ax.axvline(
                self._values[key],
                color=self.COLORS[key],
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=1.0,
            )

        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.set_title(f"Landmark adjustment — Group {self._block_idx}")
        self._refresh_visibility()
        self._refresh_marker_styles()
        self.reset_view()

    def _full_x_bounds(self):
        finite = self._time[np.isfinite(self._time)]
        if finite.size < 2:
            return None
        return float(np.min(finite)), float(np.max(finite))

    def _set_view(self, left, right):
        bounds = self._full_x_bounds()
        if bounds is None:
            return
        full_left, full_right = bounds
        left = max(full_left, float(left))
        right = min(full_right, float(right))
        if right <= left:
            return
        self.ax.set_xlim(left, right)

        # Rescale Y to the visible waveform. This makes small T-wave features
        # much easier to inspect when the large spike would otherwise dominate.
        mask = np.isfinite(self._time) & np.isfinite(self._voltage) & (self._time >= left) & (self._time <= right)
        y = self._voltage[mask]
        y = y[np.isfinite(y)]
        if self._baseline is not None and np.isfinite(self._baseline):
            y = np.append(y, float(self._baseline))
        if y.size:
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            span = ymax - ymin
            if span <= 0:
                span = max(abs(ymax) * 0.1, 0.01)
            pad = span * 0.10
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self._sync_scrollbar()
        self.canvas.draw_idle()

    def _sync_scrollbar(self):
        """Keep the horizontal scrollbar synchronized with the current x-view."""
        if not hasattr(self, "hscroll"):
            return
        bounds = self._full_x_bounds()
        if bounds is None:
            self.hscroll.setEnabled(False)
            return
        full_left, full_right = bounds
        full_span = full_right - full_left
        left, right = self.ax.get_xlim()
        view_span = max(0.0, right - left)
        if full_span <= 0 or view_span >= full_span * 0.999999:
            self._updating_scrollbar = True
            try:
                self.hscroll.setEnabled(False)
                self.hscroll.setRange(0, 10_000)
                self.hscroll.setPageStep(10_000)
                self.hscroll.setValue(0)
            finally:
                self._updating_scrollbar = False
            return

        max_offset = max(full_span - view_span, 1e-12)
        frac = min(1.0, max(0.0, (left - full_left) / max_offset))
        page = max(1, min(9_999, int(round((view_span / full_span) * 10_000))))
        self._updating_scrollbar = True
        try:
            self.hscroll.setEnabled(True)
            self.hscroll.setRange(0, 10_000)
            self.hscroll.setPageStep(page)
            self.hscroll.setValue(int(round(frac * 10_000)))
        finally:
            self._updating_scrollbar = False

    def _on_scrollbar_changed(self, value):
        if self._updating_scrollbar:
            return
        bounds = self._full_x_bounds()
        if bounds is None:
            return
        full_left, full_right = bounds
        left, right = self.ax.get_xlim()
        view_span = right - left
        full_span = full_right - full_left
        if view_span <= 0 or view_span >= full_span:
            return
        max_offset = full_span - view_span
        new_left = full_left + (float(value) / 10_000.0) * max_offset
        self._set_view(new_left, new_left + view_span)

    def reset_view(self):
        bounds = self._full_x_bounds()
        if bounds is None:
            return
        self._set_view(*bounds)

    def focus_region(self, region):
        bounds = self._full_x_bounds()
        if bounds is None:
            return
        full_left, full_right = bounds
        full_span = max(full_right - full_left, 1.0)
        if region == "spike":
            keys = ("spike_start", "spike_max", "spike_min")
            vals = [self._values[k] for k in keys if k in self._values and np.isfinite(self._values[k])]
            if not vals:
                return
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1.0)
            pad = max(8.0, span * 0.65)
            self._set_view(lo - pad, hi + pad)
        elif region == "twave":
            keys = ("t_wave_start", "t_wave_peak", "t_wave_end")
            vals = [self._values[k] for k in keys if k in self._values and np.isfinite(self._values[k])]
            if not vals:
                return
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1.0)
            pad = max(12.0, span * 0.15)
            self._set_view(lo - pad, hi + pad)
        else:
            self.reset_view()

    def focus_selected(self):
        bounds = self._full_x_bounds()
        key = self._selected_marker
        if bounds is None or key not in self._values or not np.isfinite(self._values[key]):
            return
        full_left, full_right = bounds
        full_span = max(full_right - full_left, 1.0)
        width = max(30.0, min(80.0, full_span * 0.12))
        center = float(self._values[key])
        self._set_view(center - width / 2.0, center + width / 2.0)

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        bounds = self._full_x_bounds()
        if bounds is None:
            return
        full_left, full_right = bounds
        left, right = self.ax.get_xlim()
        if right <= left:
            return
        factor = 0.78 if event.button == "up" else 1.28
        cursor = float(event.xdata)
        new_left = cursor - (cursor - left) * factor
        new_right = cursor + (right - cursor) * factor
        min_width = max((full_right - full_left) * 0.005, 2.0)
        if new_right - new_left < min_width:
            center = cursor
            new_left = center - min_width / 2.0
            new_right = center + min_width / 2.0
        if new_right - new_left >= full_right - full_left:
            self.reset_view()
        else:
            self._set_view(new_left, new_right)

    def _refresh_visibility(self):
        spectrum_visible = not self._flags.get("exclude_spectrum", False)
        if self._waveform_line is not None:
            self._waveform_line.set_alpha(0.82 if spectrum_visible else 0.08)
        for key, line in self._marker_lines.items():
            line.set_visible(bool(self._enabled.get(key, False)) and spectrum_visible)

    def _refresh_marker_styles(self):
        for key, line in self._marker_lines.items():
            selected = key == self._selected_marker
            if key == "EAD":
                line.set_linewidth(3.2 if selected else 2.5)
            else:
                line.set_linewidth(2.8 if selected else 1.8)
            line.set_alpha(1.0 if selected else 0.78)
        self._refresh_visibility()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if getattr(event, "dblclick", False):
            self.reset_view()
            return
        if event.xdata is None:
            return
        key = self._selected_marker
        if key not in self.COLORS or not self._enabled.get(key, False):
            return
        value = float(event.xdata)
        self.set_marker_value(key, value, redraw=True)
        self.markerMoved.emit(key, value)
