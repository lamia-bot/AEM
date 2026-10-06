from __future__ import annotations

import numpy as np


POLYORDER = 3


def apply_v7_smoothing(time_values, voltage_raw, filter_window_ms: float):
    """Apply the Savitzky–Golay smoothing used in v7 Cell 2.75.

    Returns (filtered_voltage, dt_ms, window_len_samples). A window length of 0
    means smoothing was disabled. The calculation intentionally follows the v7
    rules for valid finite recordings.
    """
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_raw, dtype=np.float64)

    if time.size != voltage.size or time.size < 2:
        raise ValueError("The selected range must contain at least two samples.")
    if not np.all(np.isfinite(time)):
        raise ValueError("The selected range contains non-finite time values.")
    if not np.all(np.isfinite(voltage)):
        raise ValueError(
            "The selected range contains missing/non-finite voltage values. "
            "The frozen v7 Z-score logic requires finite values."
        )

    diffs = np.diff(time)
    if diffs.size == 0 or not np.all(np.isfinite(diffs)):
        raise ValueError("Could not determine the sampling interval for this range.")
    dt_ms = float(np.median(diffs))
    if dt_ms <= 0:
        raise ValueError("Time values must increase through the selected range.")

    if float(filter_window_ms) <= 0:
        return voltage.copy(), dt_ms, 0

    window_len_samples = int(round(float(filter_window_ms) / dt_ms))
    if window_len_samples <= POLYORDER:
        window_len_samples = POLYORDER + 2
    if window_len_samples % 2 == 0:
        window_len_samples += 1

    if window_len_samples >= len(voltage):
        window_len_samples = len(voltage) - 1
        if window_len_samples % 2 == 0:
            window_len_samples -= 1

    # v7 assumes the selected range is long enough for a cubic SG filter.
    # Convert that notebook crash into a clear GUI warning rather than changing
    # the scientific method.
    if window_len_samples <= POLYORDER or window_len_samples < 1:
        raise ValueError(
            "The selected range is too short for the v7 Savitzky–Golay filter. "
            "Choose a longer range or set smoothing to 0 ms."
        )

    # SciPy is intentionally imported only when smoothing is actually used.
    # This keeps application startup lighter; once imported, Python keeps it
    # resident for the rest of the session.
    from scipy.signal import savgol_filter
    filtered = savgol_filter(voltage, window_len_samples, POLYORDER)
    return np.asarray(filtered, dtype=np.float64), dt_ms, int(window_len_samples)


def calculate_z_score_v7(voltage):
    """v7 calculate_z_score: (x - mean) / population standard deviation."""
    values = np.asarray(voltage, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No voltage data are available for Z-score calculation.")
    mean = np.mean(values)
    std_dev = np.std(values)
    if not np.isfinite(std_dev) or std_dev == 0:
        raise ValueError("The selected signal has zero or invalid standard deviation.")
    return (values - mean) / std_dev


def detect_spikes_v7(z_scores, time_values, threshold: float):
    """Scientifically equivalent vectorized form of v7 detect_spikes.

    v7 defines a spike at sample i when z[i-1] < threshold and
    z[i] >= threshold. The returned indices/times are therefore identical to
    the notebook algorithm while avoiding a Python loop over every sample.
    """
    z = np.asarray(z_scores, dtype=np.float64)
    time = np.asarray(time_values, dtype=np.float64)
    if z.size != time.size:
        raise ValueError("Z-score and time arrays must have the same length.")
    if z.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    crossings = np.flatnonzero((z[:-1] < threshold) & (z[1:] >= threshold)) + 1
    return time[crossings].astype(np.float64, copy=False), crossings.astype(np.int64, copy=False)


def divide_spikes_into_blocks_v7(spike_timepoints, averaging_factor: int):
    """Reproduce v7 Cell 3.5 block division, with the empty-input crash guarded.

    Complete blocks contain exactly `averaging_factor` detected spikes. A final
    incomplete block is discarded, matching v7.
    """
    spikes = list(np.asarray(spike_timepoints, dtype=np.float64))
    factor = int(averaging_factor)
    if factor < 1:
        raise ValueError("Averaging factor must be at least 1.")
    if not spikes:
        return {}, 0
    if factor > len(spikes):
        raise ValueError("Averaging factor cannot exceed the number of detected peaks.")

    blocks = {
        idx + 1: spikes[i : i + factor]
        for idx, i in enumerate(range(0, len(spikes), factor))
    }
    discarded = 0
    if blocks and len(list(blocks.values())[-1]) < factor:
        discarded = len(list(blocks.values())[-1])
        blocks.pop(len(blocks))
    return blocks, discarded


def compute_isi_results_v7(blocks):
    """Reproduce v7 Cell 3.5 ISI mean/std calculation for each block."""
    results = {}
    for block_idx, spike_times in blocks.items():
        if len(spike_times) < 2:
            results[int(block_idx)] = (None, None)
        else:
            isi_values = np.diff(np.asarray(spike_times, dtype=np.float64))
            results[int(block_idx)] = (float(np.mean(isi_values)), float(np.std(isi_values)))
    return results


def extract_snippets_v7(time_values, voltage_values, spike_timepoints, x_ms: float = 20.0, max_extension: float = 1700.0):
    """Reproduce the v7 `extract_snippets` calculation used in Cell 4.

    Snippets start `x_ms` before each threshold crossing and extend up to
    `max_extension` ms after it, but stop before the next snippet would begin.
    All snippets are trimmed to the shortest valid snippet exactly as in v7.
    """
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    spikes = list(np.asarray(spike_timepoints, dtype=np.float64))

    if time.size != voltage.size or time.size < 2:
        return [], np.array([], dtype=np.float64), [], 0.0
    diffs = np.diff(time)
    if diffs.size == 0:
        return [], np.array([], dtype=np.float64), [], 0.0
    dt = float(np.min(diffs))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not determine a positive sampling interval for snippet extraction.")

    pre_spike_length = int(float(x_ms) / dt)
    max_post_spike_length = int(float(max_extension) / dt)

    aligned_snippets = []
    valid_spikes = []
    snippet_lengths = []

    for i, spike_time in enumerate(spikes):
        spike_idx = int(np.searchsorted(time, spike_time))
        snippet_start_idx = spike_idx - pre_spike_length
        if snippet_start_idx < 0:
            continue

        max_end_idx = spike_idx + max_post_spike_length
        if i < len(spikes) - 1:
            next_snippet_start = int(np.searchsorted(time, spikes[i + 1] - float(x_ms)))
            max_end_idx = min(max_end_idx, next_snippet_start)
        max_end_idx = min(max_end_idx, len(time) - 1)

        snippet = voltage[snippet_start_idx:max_end_idx]
        snippet_lengths.append(len(snippet))
        aligned_snippets.append(np.asarray(snippet, dtype=np.float64))
        valid_spikes.append(float(spike_time))

    snippet_length = min(snippet_lengths) if snippet_lengths else 0
    if snippet_length <= 0:
        return [], np.array([], dtype=np.float64), [], 0.0

    snippet_times = np.linspace(
        -float(x_ms),
        snippet_length * dt - float(x_ms),
        snippet_length,
    )
    aligned_snippets = [snippet[:snippet_length] for snippet in aligned_snippets]
    return aligned_snippets, snippet_times, valid_spikes, float(snippet_length * dt)


def compute_selected_mean_v7(snippets, selected_mask=None):
    """Mean used by v7 InteractiveSnippetOverlayPlot."""
    if not snippets:
        return np.array([], dtype=np.float64)
    if selected_mask is None:
        selected_mask = np.ones(len(snippets), dtype=bool)
    mask = np.asarray(selected_mask, dtype=bool)
    selected = [s for i, s in enumerate(snippets) if i < len(mask) and mask[i]]
    if not selected:
        return np.zeros_like(np.asarray(snippets[0], dtype=np.float64))
    return np.mean(np.asarray(selected, dtype=np.float64), axis=0)


def determine_baseline_v7(time_values, voltage_values, pre_spike_fraction: float = 0.5, post_spike_window: float = 50.0):
    """Reproduce v7 Cell 5 baseline calculation for one mean waveform.

    The pre-spike baseline uses the first `pre_spike_fraction` of the portion
    before t=0. The post-spike baseline uses the final `post_spike_window` ms.
    v7 reports the region with the lower standard deviation as the more stable
    region, while its interactive selector initializes the *final* baseline at
    the pre-spike value. The GUI preserves both behaviors.
    """
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size == 0 or voltage.size == 0 or time.size != voltage.size:
        raise ValueError("Mean waveform is empty or invalid; baseline cannot be determined.")

    pre_spike_end_idx = int(np.argmin(np.abs(time)))
    pre_spike_end = float(time[pre_spike_end_idx])
    pre_spike_start = float(time[0])
    pre_spike_cutoff = pre_spike_start + float(pre_spike_fraction) * (pre_spike_end - pre_spike_start)

    pre_mask = (time >= pre_spike_start) & (time <= pre_spike_cutoff)
    if np.any(pre_mask):
        baseline_pre = float(np.mean(voltage[pre_mask]))
        std_pre = float(np.std(voltage[pre_mask]))
    else:
        baseline_pre = float("nan")
        std_pre = float("inf")

    post_spike_end = float(time[-1])
    post_spike_start = float(np.max(time) - float(post_spike_window))
    post_mask = (time >= post_spike_start) & (time <= post_spike_end)
    if np.any(post_mask):
        baseline_post = float(np.mean(voltage[post_mask]))
        std_post = float(np.std(voltage[post_mask]))
    else:
        baseline_post = float("nan")
        std_post = float("inf")

    if std_post < std_pre:
        best_baseline = baseline_post
        selected_region = "Post-Spike"
    else:
        best_baseline = baseline_pre
        selected_region = "Pre-Spike"

    return (
        float(best_baseline),
        float(baseline_pre),
        float(baseline_post),
        (float(pre_spike_start), float(pre_spike_cutoff)),
        (float(post_spike_start), float(post_spike_end)),
        selected_region,
        float(std_pre),
        float(std_post),
    )


def compute_z_score_landmarks_v7(voltage_values):
    """v7 Cell 6 Z-score helper used for mean-waveform landmark extraction."""
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if voltage.size == 0:
        return np.array([], dtype=np.float64)
    mean_val = np.mean(voltage)
    std_val = np.std(voltage)
    if std_val == 0:
        return np.zeros_like(voltage)
    return (voltage - mean_val) / std_val


def find_spike_max_min_v7(time_values, z_scores):
    """Return the times of the maximum and minimum Z-score, matching v7 Cell 6."""
    time = np.asarray(time_values, dtype=np.float64)
    z = np.asarray(z_scores, dtype=np.float64)
    if time.size == 0 or z.size == 0 or time.size != z.size:
        raise ValueError("Mean waveform is empty or invalid; spike extrema cannot be determined.")
    return float(time[int(np.argmax(z))]), float(time[int(np.argmin(z))])


def find_spike_start_v7(time_values, voltage_values, baseline: float, max_time: float):
    """Find the last baseline crossing before the positive spike peak, matching v7."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size == 0 or time.size != voltage.size:
        raise ValueError("Mean waveform is empty or invalid; spike start cannot be determined.")
    matches = np.where(time == float(max_time))[0]
    max_index = int(matches[0]) if matches.size else int(np.argmin(np.abs(time - float(max_time))))
    for i in range(max_index, 0, -1):
        if (
            (voltage[i] >= baseline and voltage[i - 1] < baseline)
            or (voltage[i] <= baseline and voltage[i - 1] > baseline)
        ):
            return float(time[i])
    return float(time[0])


def determine_noise_floor_v7(time_values, voltage_values, window_size_ms: float = 50.0):
    """Reproduce the v7 sliding-window noise-floor calculation."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size < 2 or time.size != voltage.size:
        raise ValueError("Mean waveform is too short to determine the noise floor.")
    duration = float(time[-1] - time[0])
    if duration <= 0:
        raise ValueError("Mean waveform time axis must increase.")
    window_size_samples = int((float(window_size_ms) / duration) * len(time))
    min_std = float("inf")
    best_sigma = 0.01
    for i in range(len(time) - window_size_samples):
        window_std = float(np.std(voltage[i : i + window_size_samples]))
        if window_std < min_std:
            min_std = window_std
            best_sigma = window_std
    return float(max(best_sigma, 0.01))


def find_t_wave_end_v7(time_values, voltage_values, baseline: float, noise_floor_threshold: float, noise_floor: float):
    """Reproduce v7 T-wave-end search from the end of the mean waveform."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size == 0 or time.size != voltage.size:
        raise ValueError("Mean waveform is empty or invalid; T-wave end cannot be determined.")
    threshold = float(noise_floor_threshold) * float(noise_floor)
    for i in range(len(time) - 1, 0, -1):
        if (
            (voltage[i] >= baseline and voltage[i - 1] < baseline)
            or (voltage[i] <= baseline and voltage[i - 1] > baseline)
        ):
            candidate = float(time[i])
            for j in range(i, 0, -1):
                if abs(float(voltage[j]) - float(baseline)) > threshold:
                    return candidate
    return float(time[int(len(time) * 0.8)])


def find_t_wave_start_v7(
    time_values,
    voltage_values,
    baseline: float,
    t_wave_end: float,
    min_time: float,
    fallback_offset: float,
    noise_floor_threshold: float,
    noise_floor: float,
):
    """Reproduce the refined v7 T-wave-start search."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size == 0 or time.size != voltage.size:
        raise ValueError("Mean waveform is empty or invalid; T-wave start cannot be determined.")
    threshold = float(noise_floor_threshold) * float(noise_floor)
    start_scan_idx = int(np.argmin(np.abs(time - (float(t_wave_end) - 10.0))))
    min_idx = int(np.argmin(np.abs(time - float(min_time))))

    for i in range(start_scan_idx, min_idx, -1):
        prev_voltage = voltage[i - 1] if i > 0 else voltage[i]
        if (
            (voltage[i] > baseline and prev_voltage < baseline)
            or (voltage[i] < baseline and prev_voltage > baseline)
        ):
            new_sign = np.sign(voltage[i] - baseline)
            deviation_check = any(
                new_sign * (voltage[i + j] - baseline) > threshold
                for j in range(1, min(10, len(voltage) - i))
            )
            if deviation_check:
                return float(time[i])

    return float(t_wave_end) - float(fallback_offset)


def find_t_wave_peak_v7(time_values, voltage_values, t_wave_start: float, t_wave_end: float, baseline: float):
    """Reproduce v7 T-wave peak selection between the detected start and end."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    mask = (time >= float(t_wave_start)) & (time <= float(t_wave_end))
    if np.sum(mask) == 0:
        return (float(t_wave_start) + float(t_wave_end)) / 2.0
    region = voltage[mask]
    mean_voltage = float(np.mean(region))
    if mean_voltage - float(baseline) > 0:
        peak_index = int(np.argmax(region))
    else:
        peak_index = int(np.argmin(region))
    return float(time[mask][peak_index])


def extract_landmarks_v7(
    time_values,
    voltage_values,
    baseline: float,
    window_size_ms: float = 50.0,
    noise_floor_threshold: float = 1.5,
    t_wave_fallback_start_offset: float = 30.0,
):
    """Run the exact sequence used by v7 Cell 6 for one averaged waveform."""
    time = np.asarray(time_values, dtype=np.float64)
    voltage = np.asarray(voltage_values, dtype=np.float64)
    if time.size == 0 or voltage.size == 0 or time.size != voltage.size:
        raise ValueError("Mean waveform is empty or invalid; landmarks cannot be extracted.")

    z_scores = compute_z_score_landmarks_v7(voltage)
    max_time, min_time = find_spike_max_min_v7(time, z_scores)
    spike_start = find_spike_start_v7(time, voltage, float(baseline), max_time)
    noise_floor = determine_noise_floor_v7(time, voltage, float(window_size_ms))
    t_wave_end = find_t_wave_end_v7(
        time, voltage, float(baseline), float(noise_floor_threshold), noise_floor
    )
    t_wave_start = find_t_wave_start_v7(
        time,
        voltage,
        float(baseline),
        t_wave_end,
        min_time,
        float(t_wave_fallback_start_offset),
        float(noise_floor_threshold),
        noise_floor,
    )
    t_wave_peak = find_t_wave_peak_v7(
        time, voltage, t_wave_start, t_wave_end, float(baseline)
    )
    ead_default_time = float(time[len(time) // 2])
    return {
        "spike_max": float(max_time),
        "spike_min": float(min_time),
        "spike_start": float(spike_start),
        "t_wave_start": float(t_wave_start),
        "t_wave_peak": float(t_wave_peak),
        "t_wave_end": float(t_wave_end),
        "EAD": ead_default_time,
    }, float(noise_floor)


def calculate_fpd_v7(start_time, end_time):
    """v7 Cell 8 FPD calculation: end time minus spike-start time."""
    if start_time is None or end_time is None:
        return None
    return float(end_time) - float(start_time)


def calculate_corrected_fpd_v7(fpd, isi_mean, method="fridericia"):
    """v7 Cell 8 Fridericia/Bazett correction, preserving v7 units and formula."""
    if fpd is None or isi_mean is None:
        return None
    fpd = float(fpd)
    isi_mean = float(isi_mean)
    if not np.isfinite(fpd) or not np.isfinite(isi_mean) or isi_mean <= 0:
        return None
    method = str(method).lower()
    if method == "fridericia":
        return float(fpd / (isi_mean ** (1.0 / 3.0)))
    if method == "bazett":
        return float(fpd / (isi_mean ** (1.0 / 2.0)))
    raise ValueError("Invalid correction method. Use 'fridericia' or 'bazett'.")


RESULT_METRIC_KEYS = [
    "isi_mean",
    "isi_std",
    "fpd_new",
    "fpd_traditional",
    "fpd_new_fridericia",
    "fpd_traditional_fridericia",
    "fpd_new_bazett",
    "fpd_traditional_bazett",
    "ead_time",
]


def compute_results_v7(
    landmark_values,
    landmark_enabled,
    landmark_flags,
    isi_results,
):
    """Compute the v7 Cell 8 metrics for all averaging groups.

    Marker visibility/inclusion is an explicit v8 GUI layer. The numeric marker
    positions themselves remain the v7 automatic/manual values. A metric is only
    produced when every marker it requires is enabled. This lets a hidden marker
    retain its calculated value without silently contributing to Results.
    """
    by_group = {}
    ead_count = 0

    group_ids = sorted(set(landmark_values) | set(isi_results))
    for group_idx in group_ids:
        values = dict(landmark_values.get(group_idx, {}))
        enabled = dict(landmark_enabled.get(group_idx, {}))
        flags = dict(landmark_flags.get(group_idx, {}))
        exclude_spectrum = bool(flags.get("exclude_spectrum", False))

        isi_mean, isi_std = isi_results.get(group_idx, (None, None))
        isi_mean = None if isi_mean is None else float(isi_mean)
        isi_std = None if isi_std is None else float(isi_std)

        spike_start_ok = bool(enabled.get("spike_start", False)) and "spike_start" in values
        t_peak_ok = bool(enabled.get("t_wave_peak", False)) and "t_wave_peak" in values
        t_end_ok = bool(enabled.get("t_wave_end", False)) and "t_wave_end" in values
        ead_on = bool(enabled.get("EAD", False)) and "EAD" in values

        if ead_on:
            ead_count += 1
        ead_time = float(values["EAD"]) if ead_on else None

        if exclude_spectrum:
            fpd_new = None
            fpd_traditional = None
        else:
            fpd_new = (
                calculate_fpd_v7(values.get("spike_start"), values.get("t_wave_end"))
                if spike_start_ok and t_end_ok
                else None
            )
            fpd_traditional = (
                calculate_fpd_v7(values.get("spike_start"), values.get("t_wave_peak"))
                if spike_start_ok and t_peak_ok
                else None
            )

        row = {
            "isi_mean": isi_mean,
            "isi_std": isi_std,
            "fpd_new": fpd_new,
            "fpd_traditional": fpd_traditional,
            "fpd_new_fridericia": calculate_corrected_fpd_v7(fpd_new, isi_mean, "fridericia"),
            "fpd_traditional_fridericia": calculate_corrected_fpd_v7(
                fpd_traditional, isi_mean, "fridericia"
            ),
            "fpd_new_bazett": calculate_corrected_fpd_v7(fpd_new, isi_mean, "bazett"),
            "fpd_traditional_bazett": calculate_corrected_fpd_v7(
                fpd_traditional, isi_mean, "bazett"
            ),
            "ead_time": ead_time,
            "exclude_spectrum": exclude_spectrum,
        }
        by_group[int(group_idx)] = row

    overall = {}
    for key in RESULT_METRIC_KEYS:
        vals = [row.get(key) for row in by_group.values()]
        numeric = np.asarray(
            [float(v) for v in vals if v is not None and np.isfinite(float(v))],
            dtype=np.float64,
        )
        if numeric.size:
            overall[key] = (float(np.mean(numeric)), float(np.std(numeric)))
        else:
            overall[key] = (None, None)

    return by_group, overall, int(ead_count)
