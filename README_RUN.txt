MEA Analyzer v8 - Build 22

RUN ON MAC
1. Activate the existing environment:
   source ~/opt/anaconda3/bin/activate mea
2. From Terminal/VS Code:
   cd ~/Desktop/MEA_Analyzer_v8_22
   python app.py

WHAT CHANGED IN BUILD 22
- Analysis/UI behavior from Build 21 is unchanged.
- Excel output is redesigned around one electrode per export.
- Workbook contains only Summary + E01 for a new single-electrode export.
- Summary reserves 200 electrode columns so additional electrode columns can be pasted later without changing earlier data.
- Summary has three figures: final variables Mean +/- SD across electrodes; one final averaged-beat trace per electrode; final across-electrode mean waveform with +/- 1 SD boundaries.
- Summary waveform uses a fixed, simple export rule: spike maximum = 0 ms, -100 to +1700 ms, 1 ms linear interpolation; unavailable portions are blank.
- E01 keeps the full-resolution final averaged beat, final electrode statistics, group results, and final landmarks.
- Raw selected-range data, detection arrays, and separate per-group Excel tabs are no longer duplicated into the workbook.
- Automatic append-to-existing-workbook is NOT enabled yet; Build 22 is intentionally for validating the new workbook structure first. Manual copy/paste into the next reserved Summary electrode column is supported by design.

SCIENCE
No Detection, Savitzky-Golay smoothing, Z-score threshold crossing, beat grouping, snippet selection, baseline, landmark, FPD/FPDc, ISI, or exclusion calculation was changed. The new final electrode waveform is export-only and does not feed back into scientific results.
