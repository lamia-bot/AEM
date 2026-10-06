from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


METRIC_LABELS = [
    ("isi_mean", "Inter-Spike Interval"),
    ("isi_std", "Inter-Spike Interval within-group SD"),
    ("fpd_new", "FPD New"),
    ("fpd_traditional", "FPD Traditional"),
    ("fpd_new_fridericia", "FPD New (Fridericia)"),
    ("fpd_traditional_fridericia", "FPD Traditional (Fridericia)"),
    ("fpd_new_bazett", "FPD New (Bazett)"),
    ("fpd_traditional_bazett", "FPD Traditional (Bazett)"),
    ("ead_time", "EAD Time"),
]

# Rows that represent the electrode-level mean of a scientific variable and are
# therefore suitable for the across-electrode Mean ± SD chart on Summary.
SUMMARY_PLOT_METRICS = [
    "isi_mean",
    "fpd_new",
    "fpd_traditional",
    "fpd_new_fridericia",
    "fpd_traditional_fridericia",
    "fpd_new_bazett",
    "fpd_traditional_bazett",
    "ead_time",
]

MARKER_LABELS = [
    ("spike_max", "Spike max"),
    ("spike_min", "Spike min"),
    ("spike_start", "Spike start"),
    ("t_wave_start", "T-wave start"),
    ("t_wave_peak", "T-wave peak"),
    ("t_wave_end", "T-wave end"),
    ("EAD", "EAD"),
]

MARKER_COLORS = {
    "spike_max": "#dc2626",
    "spike_min": "#991b1b",
    "spike_start": "#7e22ce",
    "t_wave_start": "#2563eb",
    "t_wave_peak": "#1e3a8a",
    "t_wave_end": "#0284c7",
    "EAD": "#111827",
}

# Build 22 export contract.  The Summary waveform is deliberately simple and
# stable so a later electrode can be pasted/appended without rewriting earlier
# columns: peak maximum at 0 ms, -100..1700 ms, one point every 1 ms.
SUMMARY_ELECTRODE_SLOTS = 200
SUMMARY_TIME_START_MS = -100.0
SUMMARY_TIME_END_MS = 1700.0
SUMMARY_TIME_STEP_MS = 1.0
SUMMARY_GRID = np.arange(
    SUMMARY_TIME_START_MS,
    SUMMARY_TIME_END_MS + 0.5 * SUMMARY_TIME_STEP_MS,
    SUMMARY_TIME_STEP_MS,
    dtype=np.float64,
)


def _finite_or_blank(value):
    if value is None:
        return ""
    try:
        f = float(value)
    except Exception:
        return value
    return f if np.isfinite(f) else ""


def _downsample_xy(x, y, max_points=10000):
    x = np.asarray(x if x is not None else [], dtype=float)
    y = np.asarray(y if y is not None else [], dtype=float)
    n = min(x.size, y.size)
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x, y = x[:n], y[:n]
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x[idx], y[idx]


def _electrode_full_resolution_waveform(state):
    """Return one final full-resolution averaged beat for the electrode.

    This is export-only and does not alter any v7 calculation.  It pools every
    user-accepted beat from non-excluded averaging groups, trims them to the
    shortest accepted beat (the same pointwise-mean principle used by v7 within
    a group), and takes a simple arithmetic mean.  If every group was marked
    excluded, it falls back to the accepted beats from all groups so the export
    still contains a trace.
    """
    selected = []
    selected_time_candidates = []

    def collect(skip_excluded: bool):
        traces = []
        times = []
        for group_idx in sorted(state.averaging_snippets):
            if skip_excluded and state.landmark_flags.get(group_idx, {}).get("exclude_spectrum", False):
                continue
            snippets = state.averaging_snippets.get(group_idx, [])
            mask = np.asarray(state.averaging_selected_masks.get(group_idx, []), dtype=bool)
            group_time = np.asarray(state.averaging_snippet_times.get(group_idx, []), dtype=np.float64)
            for i, snippet in enumerate(snippets):
                if i >= mask.size or not bool(mask[i]):
                    continue
                arr = np.asarray(snippet, dtype=np.float64)
                if arr.size:
                    traces.append(arr)
                    times.append(group_time)
        return traces, times

    selected, selected_time_candidates = collect(skip_excluded=True)
    if not selected:
        selected, selected_time_candidates = collect(skip_excluded=False)
    if not selected:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0

    min_len = min(arr.size for arr in selected)
    if min_len <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0
    stack = np.asarray([arr[:min_len] for arr in selected], dtype=np.float64)
    mean = np.mean(stack, axis=0)

    # All snippets originate from the same recording and v7 relative-time
    # construction. Use the first matching time axis and trim it identically.
    time = np.asarray(selected_time_candidates[0], dtype=np.float64)[:min_len]
    if time.size != mean.size:
        n = min(time.size, mean.size)
        time, mean = time[:n], mean[:n]
    return time, mean, len(selected)


def _summary_waveform(full_time, full_voltage):
    """Peak-align one electrode and linearly resample it to the fixed Summary grid."""
    time = np.asarray(full_time, dtype=np.float64)
    voltage = np.asarray(full_voltage, dtype=np.float64)
    n = min(time.size, voltage.size)
    if n < 2:
        return np.full(SUMMARY_GRID.shape, np.nan, dtype=np.float64), None
    time, voltage = time[:n], voltage[:n]
    finite = np.isfinite(time) & np.isfinite(voltage)
    time, voltage = time[finite], voltage[finite]
    if time.size < 2:
        return np.full(SUMMARY_GRID.shape, np.nan, dtype=np.float64), None

    # v7 spike_max is argmax of the Z-score, equivalent to argmax of voltage.
    peak_idx = int(np.argmax(voltage))
    peak_time = float(time[peak_idx])
    aligned_time = time - peak_time

    # np.interp cannot emit NaN outside the source interval unless asked to.
    out = np.interp(SUMMARY_GRID, aligned_time, voltage, left=np.nan, right=np.nan)
    return out, peak_time


def _accepted_beat_count(state, exclude_flagged=True):
    total = 0
    for group_idx, mask in state.averaging_selected_masks.items():
        if exclude_flagged and state.landmark_flags.get(group_idx, {}).get("exclude_spectrum", False):
            continue
        total += int(np.sum(np.asarray(mask, dtype=bool)))
    return total


def _landmark_plot_png(state, group_idx: int) -> BytesIO:
    time = np.asarray(state.averaging_snippet_times.get(group_idx, []), dtype=float)
    mean = np.asarray(state.averaging_mean_snippets.get(group_idx, []), dtype=float)
    baseline = state.baseline_values.get(group_idx)
    values = state.landmark_values.get(group_idx, {})
    enabled = state.landmark_enabled.get(group_idx, {})
    flags = state.landmark_flags.get(group_idx, {})

    fig = Figure(figsize=(8.5, 4.4), dpi=150, tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    if time.size and mean.size:
        ax.plot(time, mean, color="#111827", linewidth=1.2, alpha=0.88, label="Mean waveform")
    if baseline is not None and np.isfinite(float(baseline)):
        ax.axhline(float(baseline), color="#6b7280", linewidth=1.1, alpha=0.7, label="Baseline")
    if not flags.get("exclude_spectrum", False):
        for key, label in MARKER_LABELS:
            if enabled.get(key, False) and key in values:
                ax.axvline(
                    float(values[key]),
                    color=MARKER_COLORS[key],
                    linestyle="-" if key == "EAD" else "--",
                    linewidth=2.1 if key == "EAD" else 1.6,
                    label=label,
                )
    ax.set_title(f"Final landmarks - Group {group_idx}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=7)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def _detection_plot_png(state) -> BytesIO:
    t, z = _downsample_xy(state.range_time, state.detection_z_scores, 12000)
    fig = Figure(figsize=(8.5, 4.0), dpi=150, tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    if t.size:
        ax.plot(t, z, linewidth=0.8, color="#2563eb")
    if state.threshold is not None:
        ax.axhline(float(state.threshold), color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title("Z-score spike detection")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Z-score")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def _summary_metric_rows():
    """Stable Summary rows: electrode values plus within-electrode SD where useful."""
    rows = [
        ("Detected peaks", "meta_detected"),
        ("Accepted beats", "meta_accepted"),
        ("Averaging factor", "meta_factor"),
        ("Analyzed groups", "meta_groups"),
        ("EAD count", "meta_ead_count"),
    ]
    for key, label in METRIC_LABELS:
        rows.append((f"{label} - electrode mean", f"metric_mean:{key}"))
        # results_overall stores the across-group SD for the electrode.
        rows.append((f"{label} - within-electrode SD", f"metric_sd:{key}"))
    return rows


def _metric_value_for_state(state, token):
    if token == "meta_detected":
        return len(state.spike_timepoints) if state.spike_timepoints is not None else 0
    if token == "meta_accepted":
        return _accepted_beat_count(state, exclude_flagged=True)
    if token == "meta_factor":
        return state.averaging_factor
    if token == "meta_groups":
        return len(state.results_by_group)
    if token == "meta_ead_count":
        return state.results_ead_count
    if token.startswith("metric_mean:"):
        return state.results_overall.get(token.split(":", 1)[1], (None, None))[0]
    if token.startswith("metric_sd:"):
        return state.results_overall.get(token.split(":", 1)[1], (None, None))[1]
    return None


def _write_summary_sheet(workbook, state, formats, full_time, full_voltage, summary_voltage):
    header = formats["header"]
    subheader = formats["subheader"]
    label_fmt = formats["label"]
    num_fmt = formats["num"]
    note_fmt = formats["note"]
    title_fmt = formats["title"]
    section_fmt = formats["section"]
    id_fmt = formats["id"]

    ws = workbook.add_worksheet("Summary")
    ws.hide_gridlines(2)
    ws.set_tab_color("#2563EB")
    ws.set_column("A:A", 42)
    ws.set_column("B:D", 16)

    first_electrode_col = 4  # E
    last_electrode_col = first_electrode_col + SUMMARY_ELECTRODE_SLOTS - 1
    first_electrode_excel = xl_col_to_name(first_electrode_col)
    last_electrode_excel = xl_col_to_name(last_electrode_col)
    ws.set_column(first_electrode_col, last_electrode_col, 13)

    ws.write("A1", "MEA Electrode Summary", title_fmt)
    ws.write("A2", "One electrode = one column. Copy/paste one complete electrode column into the next empty slot; the formulas and figures are pre-wired for 200 electrodes.", note_fmt)
    ws.write("A3", "Summary waveforms: spike maximum aligned to 0 ms; fixed -100 to +1700 ms axis; 1 ms linear interpolation; unavailable portions stay blank.", note_fmt)

    # Three figures live at the top so this sheet opens as a results dashboard.
    # Their data tables are lower on the sheet.

    metric_start_row = 41  # Excel row 42
    ws.write(metric_start_row, 0, "FINAL ELECTRODE RESULTS", section_fmt)
    ws.write(metric_start_row + 1, 0, "Parameter", header)
    ws.write(metric_start_row + 1, 1, "Across-electrode Mean", header)
    ws.write(metric_start_row + 1, 2, "Across-electrode SD", header)
    ws.write(metric_start_row + 1, 3, "n electrodes", header)
    for slot in range(SUMMARY_ELECTRODE_SLOTS):
        col = first_electrode_col + slot
        ws.write(metric_start_row + 1, col, f"E{slot + 1:02d}", header)

    # Identification rows belong to the electrode packet but are not averaged.
    id_row = metric_start_row + 2
    source_row = metric_start_row + 3
    analysis_row = metric_start_row + 4
    ws.write(id_row, 0, "Electrode ID", label_fmt)
    ws.write(source_row, 0, "Source recording", label_fmt)
    ws.write(analysis_row, 0, "Analysis name", label_fmt)
    ws.write(id_row, first_electrode_col, state.selected_electrode or "", id_fmt)
    ws.write(source_row, first_electrode_col, state.recording.source_path.name if state.recording.source_path else "")
    ws.write(analysis_row, first_electrode_col, state.analysis_name or "")

    metric_rows = {}
    row = metric_start_row + 6
    for label, token in _summary_metric_rows():
        metric_rows[token] = row
        ws.write(row, 0, label, label_fmt if token.startswith("meta_") else None)
        value = _metric_value_for_state(state, token)
        ws.write(row, first_electrode_col, _finite_or_blank(value), num_fmt if isinstance(_finite_or_blank(value), float) else None)

        rng = f"${first_electrode_excel}${row + 1}:${last_electrode_excel}${row + 1}"
        ws.write_formula(row, 1, f'=IF(COUNT({rng})=0,"",AVERAGE({rng}))', num_fmt)
        ws.write_formula(row, 2, f'=IF(COUNT({rng})>1,STDEV.S({rng}),0)', num_fmt)
        ws.write_formula(row, 3, f"=COUNT({rng})")
        row += 1

    # A compact metric table for the first chart, using only electrode-level means.
    chart_metric_start = row + 2
    ws.write_row(chart_metric_start, 0, ["Variable", "Mean", "SD"], subheader)
    plot_rows = []
    labels_by_key = dict(METRIC_LABELS)
    for i, key in enumerate(SUMMARY_PLOT_METRICS, start=chart_metric_start + 1):
        source_metric_row = metric_rows[f"metric_mean:{key}"]
        label = labels_by_key[key]
        plot_rows.append(i)
        ws.write(i, 0, label)
        ws.write_formula(i, 1, f"=B{source_metric_row + 1}", num_fmt)
        ws.write_formula(i, 2, f"=C{source_metric_row + 1}", num_fmt)

    waveform_start_row = chart_metric_start + len(SUMMARY_PLOT_METRICS) + 4
    ws.write(waveform_start_row, 0, "PEAK-ALIGNED ELECTRODE WAVEFORMS", section_fmt)
    waveform_header_row = waveform_start_row + 1
    ws.write_row(waveform_header_row, 0, ["Time (ms)", "Across-electrode Mean", "Across-electrode SD", "n electrodes"], header)
    for slot in range(SUMMARY_ELECTRODE_SLOTS):
        col = first_electrode_col + slot
        # Series name points to the Electrode ID row when charted, but a stable slot
        # label here makes manual copy/paste unambiguous even before an ID is added.
        ws.write(waveform_header_row, col, f"E{slot + 1:02d}", header)

    upper_col = last_electrode_col + 1
    lower_col = last_electrode_col + 2
    ws.write(waveform_header_row, upper_col, "Mean + SD", header)
    ws.write(waveform_header_row, lower_col, "Mean - SD", header)
    ws.set_column(upper_col, lower_col, 13, None, {"hidden": True})

    for i, t in enumerate(SUMMARY_GRID):
        r = waveform_header_row + 1 + i
        ws.write_number(r, 0, float(t), num_fmt)
        val = summary_voltage[i]
        if np.isfinite(val):
            ws.write_number(r, first_electrode_col, float(val), num_fmt)
        rng = f"${first_electrode_excel}${r + 1}:${last_electrode_excel}${r + 1}"
        ws.write_formula(r, 1, f'=IF(COUNT({rng})=0,"",AVERAGE({rng}))', num_fmt)
        ws.write_formula(r, 2, f'=IF(COUNT({rng})>1,STDEV.S({rng}),0)', num_fmt)
        ws.write_formula(r, 3, f"=COUNT({rng})")
        ws.write_formula(r, upper_col, f'=IF(B{r + 1}="","",B{r + 1}+C{r + 1})', num_fmt)
        ws.write_formula(r, lower_col, f'=IF(B{r + 1}="","",B{r + 1}-C{r + 1})', num_fmt)

    waveform_first_data_row = waveform_header_row + 1
    waveform_last_data_row = waveform_header_row + len(SUMMARY_GRID)

    # Figure 1: electrode-level mean ± SD for every final variable.
    chart1 = workbook.add_chart({"type": "column"})
    chart1.add_series({
        "name": "Mean across electrodes",
        "categories": ["Summary", plot_rows[0], 0, plot_rows[-1], 0],
        "values": ["Summary", plot_rows[0], 1, plot_rows[-1], 1],
        "y_error_bars": {
            "type": "custom",
            "plus_values": ["Summary", plot_rows[0], 2, plot_rows[-1], 2],
            "minus_values": ["Summary", plot_rows[0], 2, plot_rows[-1], 2],
        },
    })
    chart1.set_title({"name": "Final variables — Mean ± SD across electrodes"})
    chart1.set_y_axis({"name": "Value"})
    chart1.set_legend({"none": True})
    chart1.set_size({"width": 670, "height": 320})
    ws.insert_chart("A5", chart1)

    # Figure 2: one standardized final trace per electrode.  All 200 series are
    # pre-wired; blank columns simply do not draw until data are pasted/appended.
    chart2 = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    for slot in range(SUMMARY_ELECTRODE_SLOTS):
        col = first_electrode_col + slot
        chart2.add_series({
            "name": ["Summary", id_row, col],
            "categories": ["Summary", waveform_first_data_row, 0, waveform_last_data_row, 0],
            "values": ["Summary", waveform_first_data_row, col, waveform_last_data_row, col],
            "line": {"width": 1.0, "transparency": 35},
        })
    chart2.add_series({
        "name": "Mean",
        "categories": ["Summary", waveform_first_data_row, 0, waveform_last_data_row, 0],
        "values": ["Summary", waveform_first_data_row, 1, waveform_last_data_row, 1],
        "line": {"width": 2.25, "color": "#111827"},
    })
    chart2.set_title({"name": "Final averaged beat — one trace per electrode"})
    chart2.set_x_axis({"name": "Peak-aligned time (ms)", "min": SUMMARY_TIME_START_MS, "max": SUMMARY_TIME_END_MS})
    chart2.set_y_axis({"name": "Voltage (mV)"})
    chart2.set_legend({"none": True})
    chart2.set_size({"width": 670, "height": 320})
    ws.insert_chart("J5", chart2)

    # Figure 3: final average across electrodes with simple ±1 SD boundary lines.
    chart3 = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    chart3.add_series({
        "name": "Mean",
        "categories": ["Summary", waveform_first_data_row, 0, waveform_last_data_row, 0],
        "values": ["Summary", waveform_first_data_row, 1, waveform_last_data_row, 1],
        "line": {"width": 2.2, "color": "#111827"},
    })
    chart3.add_series({
        "name": "+1 SD",
        "categories": ["Summary", waveform_first_data_row, 0, waveform_last_data_row, 0],
        "values": ["Summary", waveform_first_data_row, upper_col, waveform_last_data_row, upper_col],
        "line": {"width": 1.0, "dash_type": "dash", "color": "#6B7280"},
    })
    chart3.add_series({
        "name": "-1 SD",
        "categories": ["Summary", waveform_first_data_row, 0, waveform_last_data_row, 0],
        "values": ["Summary", waveform_first_data_row, lower_col, waveform_last_data_row, lower_col],
        "line": {"width": 1.0, "dash_type": "dash", "color": "#6B7280"},
    })
    chart3.set_title({"name": "Final average across electrodes"})
    chart3.set_x_axis({"name": "Peak-aligned time (ms)", "min": SUMMARY_TIME_START_MS, "max": SUMMARY_TIME_END_MS})
    chart3.set_y_axis({"name": "Voltage (mV)"})
    chart3.set_legend({"position": "bottom"})
    chart3.set_size({"width": 670, "height": 320})
    ws.insert_chart("A21", chart3)

    ws.freeze_panes(metric_start_row + 2, first_electrode_col)
    ws.autofilter(metric_start_row + 1, 0, metric_start_row + 1 + 3 + len(_summary_metric_rows()), last_electrode_col)
    return {
        "id_row": id_row,
        "metric_rows": metric_rows,
        "waveform_header_row": waveform_header_row,
        "waveform_first_data_row": waveform_first_data_row,
        "waveform_last_data_row": waveform_last_data_row,
        "first_electrode_col": first_electrode_col,
    }


def _write_electrode_sheet(workbook, state, formats, full_time, full_voltage, accepted_count, peak_time, slot_number=1):
    header = formats["header"]
    subheader = formats["subheader"]
    label_fmt = formats["label"]
    num_fmt = formats["num"]
    note_fmt = formats["note"]
    title_fmt = formats["title"]
    section_fmt = formats["section"]

    sheet_name = f"E{slot_number:02d}"
    ws = workbook.add_worksheet(sheet_name)
    ws.hide_gridlines(2)
    ws.set_column("A:A", 28)
    ws.set_column("B:C", 18)
    ws.set_column("D:R", 16)

    ws.write("A1", f"{sheet_name} — Electrode analysis", title_fmt)
    meta = [
        ("Electrode ID", state.selected_electrode or ""),
        ("Analysis name", state.analysis_name or ""),
        ("Source recording", state.recording.source_path.name if state.recording.source_path else ""),
        ("Detected peaks", len(state.spike_timepoints) if state.spike_timepoints is not None else 0),
        ("Accepted beats in final trace", accepted_count),
        ("Averaging factor", state.averaging_factor),
        ("Analyzed groups", len(state.results_by_group)),
        ("EAD count", state.results_ead_count),
    ]
    row = 2
    for label, value in meta:
        ws.write(row, 0, label, label_fmt)
        clean = _finite_or_blank(value)
        ws.write(row, 1, clean, num_fmt if isinstance(clean, float) else None)
        row += 1

    row += 1
    ws.write(row, 0, "FINAL ELECTRODE STATISTICS", section_fmt)
    row += 1
    ws.write_row(row, 0, ["Parameter", "Mean", "Within-electrode SD"], header)
    row += 1
    for key, label in METRIC_LABELS:
        mean, std = state.results_overall.get(key, (None, None))
        ws.write(row, 0, label)
        ws.write(row, 1, _finite_or_blank(mean), num_fmt)
        ws.write(row, 2, _finite_or_blank(std), num_fmt)
        row += 1

    row += 2
    ws.write(row, 0, "GROUP RESULTS + FINAL LANDMARKS", section_fmt)
    row += 1
    group_header_row = row
    group_headers = [
        "Group", "Included beats", "Spectrum excluded",
        "ISI mean", "ISI SD", "FPD New", "FPD Traditional",
        "New Fridericia", "Traditional Fridericia", "New Bazett", "Traditional Bazett",
        "EAD time", "Baseline", "Spike start", "Spike max", "Spike min",
        "T-wave start", "T-wave peak", "T-wave end",
    ]
    ws.write_row(group_header_row, 0, group_headers, header)
    row += 1
    for group_idx in sorted(state.results_by_group):
        metrics = state.results_by_group[group_idx]
        mask = np.asarray(state.averaging_selected_masks.get(group_idx, []), dtype=bool)
        values = state.landmark_values.get(group_idx, {})
        enabled = state.landmark_enabled.get(group_idx, {})
        flags = state.landmark_flags.get(group_idx, {})
        ead_value = values.get("EAD") if enabled.get("EAD", False) else None
        data = [
            f"B{group_idx:02d}",
            int(np.sum(mask)) if mask.size else 0,
            "Yes" if flags.get("exclude_spectrum", False) else "No",
            metrics.get("isi_mean"), metrics.get("isi_std"), metrics.get("fpd_new"), metrics.get("fpd_traditional"),
            metrics.get("fpd_new_fridericia"), metrics.get("fpd_traditional_fridericia"),
            metrics.get("fpd_new_bazett"), metrics.get("fpd_traditional_bazett"),
            ead_value, state.baseline_values.get(group_idx),
            values.get("spike_start"), values.get("spike_max"), values.get("spike_min"),
            values.get("t_wave_start"), values.get("t_wave_peak"), values.get("t_wave_end"),
        ]
        for col, value in enumerate(data):
            clean = _finite_or_blank(value)
            ws.write(row, col, clean, num_fmt if isinstance(clean, float) else None)
        row += 1

    waveform_start = row + 2
    ws.write(waveform_start, 0, "FULL-RESOLUTION FINAL AVERAGED BEAT", section_fmt)
    ws.write(waveform_start + 1, 0, "This trace is the arithmetic mean of all accepted beats from non-excluded groups. It is export-only and does not change the v7 calculations.", note_fmt)
    ws.write_row(waveform_start + 3, 0, ["Snippet time (ms)", "Peak-aligned time (ms)", "Mean voltage (mV)"], header)
    for i, (t, v) in enumerate(zip(full_time, full_voltage), start=waveform_start + 4):
        ws.write_number(i, 0, float(t), num_fmt)
        ws.write_number(i, 1, float(t - peak_time) if peak_time is not None else float(t), num_fmt)
        ws.write_number(i, 2, float(v), num_fmt)

    # Keep the electrode tab light visually: the chart uses the same fixed Summary
    # representation, while the full-resolution numeric data remain directly below.
    n = len(full_time)
    if n > 1:
        chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
        chart.add_series({
            "name": "Final averaged beat",
            "categories": [sheet_name, waveform_start + 4, 1, waveform_start + 3 + n, 1],
            "values": [sheet_name, waveform_start + 4, 2, waveform_start + 3 + n, 2],
            "line": {"width": 1.35, "color": "#111827"},
        })
        chart.set_title({"name": "Full-resolution final averaged beat"})
        chart.set_x_axis({"name": "Peak-aligned time (ms)"})
        chart.set_y_axis({"name": "Voltage (mV)"})
        chart.set_legend({"none": True})
        chart.set_size({"width": 720, "height": 340})
        ws.insert_chart("E2", chart)

    ws.freeze_panes(group_header_row + 1, 1)
    return sheet_name


def export_excel(state, path: Path, progress: Optional[Callable[[int, str], None]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(10, "Preparing final electrode waveform...")

    full_time, full_voltage, accepted_count = _electrode_full_resolution_waveform(state)
    if full_time.size < 2 or full_voltage.size < 2:
        raise ValueError("No accepted waveform data are available for Excel export.")
    summary_voltage, peak_time = _summary_waveform(full_time, full_voltage)
    if peak_time is None:
        raise ValueError("Could not determine the final averaged-beat spike maximum for export.")

    if progress:
        progress(18, "Creating compact Summary workbook...")

    workbook = xlsxwriter.Workbook(str(path))
    try:
        try:
            workbook.set_calc_mode("auto")
        except Exception:
            pass

        formats = {
            "header": workbook.add_format({
                "bold": True, "font_color": "#FFFFFF", "bg_color": "#1D4ED8",
                "border": 1, "align": "center", "valign": "vcenter",
            }),
            "subheader": workbook.add_format({"bold": True, "bg_color": "#E5E7EB", "border": 1}),
            "label": workbook.add_format({"bold": True, "font_color": "#374151"}),
            "num": workbook.add_format({"num_format": "0.0000"}),
            "note": workbook.add_format({"font_color": "#6B7280", "italic": True, "text_wrap": True}),
            "title": workbook.add_format({"bold": True, "font_size": 16, "font_color": "#111827"}),
            "section": workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1D4ED8"}),
            "id": workbook.add_format({"bold": True, "font_color": "#111827", "bg_color": "#DBEAFE"}),
        }

        _write_summary_sheet(workbook, state, formats, full_time, full_voltage, summary_voltage)
        if progress:
            progress(48, "Writing full-resolution electrode sheet...")
        _write_electrode_sheet(
            workbook, state, formats, full_time, full_voltage,
            accepted_count=accepted_count, peak_time=peak_time, slot_number=1,
        )
        if progress:
            progress(62, "Finalizing Excel workbook...")
    finally:
        workbook.close()

def export_pdf(state, path: Path, progress: Optional[Callable[[int, str], None]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(65, "Creating PDF report...")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MEAReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=12,
    )
    h2 = ParagraphStyle("MEAReportH2", parent=styles["Heading2"], fontSize=14, spaceBefore=8, spaceAfter=6)
    body = styles["BodyText"]
    small = ParagraphStyle("MEASmall", parent=body, fontSize=8, textColor=colors.HexColor("#6B7280"))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=state.analysis_name or "MEA Analysis",
        author="MEA Analyzer",
    )
    story = [Paragraph(state.analysis_name or "MEA Analysis", title_style)]

    source = state.recording.source_path.name if state.recording.source_path else ""
    selected_range = state.selected_range or (None, None)
    meta_data = [
        ["Source recording", source],
        ["Electrode", state.selected_electrode or ""],
        ["Selected range (ms)", f"{selected_range[0]:.4f} to {selected_range[1]:.4f}" if selected_range[0] is not None else ""],
        ["Smoothing window (ms)", str(state.filter_window_ms)],
        ["Z-score threshold", "" if state.threshold is None else f"{state.threshold:.4g}"],
        ["Detected peaks", str(len(state.spike_timepoints) if state.spike_timepoints is not None else 0)],
        ["Averaging factor", str(state.averaging_factor or "")],
        ["Groups", str(len(state.results_by_group))],
    ]
    meta_table = Table(meta_data, colWidths=[1.8 * inch, 5.7 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [meta_table, Spacer(1, 10), Paragraph("Overall results", h2)]

    summary_data = [["Parameter", "Mean", "SD"]]
    for key, label in METRIC_LABELS:
        mean, std = state.results_overall.get(key, (None, None))
        summary_data.append([
            label,
            "" if mean is None else f"{mean:.4f}",
            "" if std is None else f"{std:.4f}",
        ])
    summary_data += [
        ["No. of EADs included", str(state.results_ead_count), ""],
        ["Total no. of groups", str(len(state.results_by_group)), ""],
        ["EAD occurrence (%)", f"{(state.results_ead_count / max(1, len(state.results_by_group)) * 100):.2f}", ""],
    ]
    summary_table = Table(summary_data, colWidths=[3.7 * inch, 1.6 * inch, 1.6 * inch], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(PageBreak())
    story.append(Paragraph("Spike detection", h2))
    detection_png = _detection_plot_png(state)
    story.append(Image(detection_png, width=7.1 * inch, height=3.35 * inch))
    story.append(Paragraph("Underlying data: Excel -> E01", small))

    group_ids = sorted(state.results_by_group)
    for pos, group_idx in enumerate(group_ids, start=1):
        story.append(PageBreak())
        block_id = f"B{group_idx:02d}"
        story.append(Paragraph(f"{block_id} - Group {group_idx}", h2))
        metrics = state.results_by_group[group_idx]
        group_table_data = [["Parameter", "Value"]]
        for key, label in METRIC_LABELS:
            val = metrics.get(key)
            group_table_data.append([label, "" if val is None else f"{float(val):.4f}"])
        group_table = Table(group_table_data, colWidths=[4.5 * inch, 2.0 * inch], repeatRows=1)
        group_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(group_table)
        story.append(Spacer(1, 8))
        lm_png = _landmark_plot_png(state, group_idx)
        story.append(Image(lm_png, width=7.1 * inch, height=3.65 * inch))
        story.append(Paragraph("Underlying data: Excel -> E01", small))
        if progress:
            pct = 68 + int(27 * pos / max(1, len(group_ids)))
            progress(pct, f"Writing PDF group {pos}/{len(group_ids)}...")

    doc.build(story)


def export_results(state, excel_path: Path, pdf_path: Path, progress: Optional[Callable[[int, str], None]] = None) -> None:
    export_excel(state, excel_path, progress)
    export_pdf(state, pdf_path, progress)
    if progress:
        progress(100, "Results saved.")
