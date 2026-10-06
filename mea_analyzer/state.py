from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class StepState(str, Enum):
    NOT_STARTED = "not_started"
    COMPLETE = "complete"
    NEEDS_UPDATE = "needs_update"
    SKIPPED = "skipped"


ANALYSIS_STEPS = [
    "range",
    "detection",
    "averaging",
    "baseline",
    "landmarks",
    "results",
]


@dataclass
class RecordingInfo:
    source_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    cache_parquet: Optional[Path] = None
    headers: list[str] = field(default_factory=list)
    time_column: Optional[str] = None
    electrode_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    row_count_exact: bool = True
    source_size: int = 0
    source_format: str = ""
    csv_delimiter: Optional[str] = None
    time_min: Optional[float] = None
    time_max: Optional[float] = None
    preview_time: Optional[object] = None
    preview_signals: dict[str, object] = field(default_factory=dict)
    electrode_has_data: dict[str, bool] = field(default_factory=dict)


@dataclass
class AnalysisState:
    recording: RecordingInfo = field(default_factory=RecordingInfo)
    selected_electrode: Optional[str] = None
    output_dir: Optional[Path] = None
    analysis_name: str = ""
    selected_range: Optional[tuple[float, float]] = None
    threshold: Optional[float] = None
    filter_window_ms: float = 10.0
    averaging_factor: Optional[int] = None

    range_time: Optional[object] = None
    range_voltage_raw: Optional[object] = None
    detection_voltage_filtered: Optional[object] = None
    detection_z_scores: Optional[object] = None
    spike_timepoints: Optional[object] = None
    spike_indices: Optional[object] = None
    detection_window_samples: int = 0
    detection_dt_ms: Optional[float] = None

    averaging_blocks: dict[int, list[float]] = field(default_factory=dict)
    averaging_isi_results: dict[int, tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    averaging_snippets: dict[int, list[object]] = field(default_factory=dict)
    averaging_snippet_times: dict[int, object] = field(default_factory=dict)
    averaging_valid_spikes: dict[int, list[float]] = field(default_factory=dict)
    averaging_selected_masks: dict[int, object] = field(default_factory=dict)
    averaging_mean_snippets: dict[int, object] = field(default_factory=dict)
    averaging_snippet_lengths_ms: dict[int, float] = field(default_factory=dict)

    baseline_values: dict[int, float] = field(default_factory=dict)
    baseline_pre: dict[int, float] = field(default_factory=dict)
    baseline_post: dict[int, float] = field(default_factory=dict)
    baseline_best: dict[int, float] = field(default_factory=dict)
    baseline_selected_region: dict[int, str] = field(default_factory=dict)
    baseline_region_pre: dict[int, tuple[float, float]] = field(default_factory=dict)
    baseline_region_post: dict[int, tuple[float, float]] = field(default_factory=dict)
    baseline_std_pre: dict[int, float] = field(default_factory=dict)
    baseline_std_post: dict[int, float] = field(default_factory=dict)

    landmark_values: dict[int, dict[str, float]] = field(default_factory=dict)
    landmark_auto_values: dict[int, dict[str, float]] = field(default_factory=dict)
    landmark_noise_floor: dict[int, float] = field(default_factory=dict)
    landmark_flags: dict[int, dict[str, bool]] = field(default_factory=dict)
    # Marker inclusion/visibility is intentionally separate from its calculated
    # numeric value. Hidden markers keep their automatic/manual position so that
    # turning them back on restores the same value immediately.
    landmark_enabled: dict[int, dict[str, bool]] = field(default_factory=dict)

    results_by_group: dict[int, dict[str, object]] = field(default_factory=dict)
    results_overall: dict[str, tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    results_ead_count: int = 0
    results_exported: bool = False
    results_excel_path: Optional[Path] = None
    results_pdf_path: Optional[Path] = None

    step_states: dict[str, StepState] = field(
        default_factory=lambda: {step: StepState.NOT_STARTED for step in ANALYSIS_STEPS}
    )

    def reset_all(self) -> None:
        """Clear the entire current analysis while preserving this state object's identity.

        Keeping the same AnalysisState instance is safer for long-lived GUI pages and
        callbacks that may retain references across a Restart / Start next file cycle.
        """
        self.recording = RecordingInfo()
        self.output_dir = None
        self.analysis_name = ""
        self.reset_for_new_recording()

    def reset_for_new_recording(self) -> None:
        self.selected_electrode = None
        self.selected_range = None
        self.threshold = None
        self.filter_window_ms = 10.0
        self.averaging_factor = None
        self.range_time = None
        self.range_voltage_raw = None
        self.detection_voltage_filtered = None
        self.detection_z_scores = None
        self.spike_timepoints = None
        self.spike_indices = None
        self.detection_window_samples = 0
        self.detection_dt_ms = None
        self.averaging_blocks = {}
        self.averaging_isi_results = {}
        self.averaging_snippets = {}
        self.averaging_snippet_times = {}
        self.averaging_valid_spikes = {}
        self.averaging_selected_masks = {}
        self.averaging_mean_snippets = {}
        self.averaging_snippet_lengths_ms = {}
        self.baseline_values = {}
        self.baseline_pre = {}
        self.baseline_post = {}
        self.baseline_best = {}
        self.baseline_selected_region = {}
        self.baseline_region_pre = {}
        self.baseline_region_post = {}
        self.baseline_std_pre = {}
        self.baseline_std_post = {}
        self.landmark_values = {}
        self.landmark_auto_values = {}
        self.landmark_noise_floor = {}
        self.landmark_flags = {}
        self.landmark_enabled = {}
        self.clear_results()
        self.step_states = {step: StepState.NOT_STARTED for step in ANALYSIS_STEPS}

    def clear_results(self) -> None:
        self.results_by_group = {}
        self.results_overall = {}
        self.results_ead_count = 0
        self.results_exported = False
        self.results_excel_path = None
        self.results_pdf_path = None

    def invalidate_from(self, step: str) -> None:
        """Mark a changed step complete and downstream completed steps as needing update."""
        if step not in ANALYSIS_STEPS:
            return
        idx = ANALYSIS_STEPS.index(step)
        self.step_states[step] = StepState.COMPLETE
        if step != "results":
            self.clear_results()
        for downstream in ANALYSIS_STEPS[idx + 1 :]:
            if self.step_states[downstream] in {StepState.COMPLETE, StepState.NEEDS_UPDATE}:
                self.step_states[downstream] = StepState.NEEDS_UPDATE

    def mark_step_dirty(self, step: str) -> None:
        """Mark a completed step and any completed downstream steps as needing update."""
        if step not in ANALYSIS_STEPS:
            return
        idx = ANALYSIS_STEPS.index(step)
        if step != "results":
            self.clear_results()
        if self.step_states[step] in {StepState.COMPLETE, StepState.NEEDS_UPDATE}:
            self.step_states[step] = StepState.NEEDS_UPDATE
        for downstream in ANALYSIS_STEPS[idx + 1 :]:
            if self.step_states[downstream] in {StepState.COMPLETE, StepState.NEEDS_UPDATE}:
                self.step_states[downstream] = StepState.NEEDS_UPDATE

    def invalidate_all_analysis(self) -> None:
        self.clear_results()
        for step in ANALYSIS_STEPS:
            if self.step_states[step] == StepState.COMPLETE:
                self.step_states[step] = StepState.NEEDS_UPDATE
            elif self.step_states[step] != StepState.SKIPPED:
                self.step_states[step] = StepState.NOT_STARTED
