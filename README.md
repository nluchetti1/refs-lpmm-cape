# Space Coast REFS QPF

48-h REFS QPF for Cape Canaveral / KSC: LPMM time-lagged trend, EAS exceedance
probabilities, LPMM vs AVRG, flash flood / recurrence interval, a pad timeline,
and running verification against MRMS. Rebuilt 4x daily by GitHub Actions.

## Setup
1. New repo; push everything here (including `.github/`).
2. Actions tab → enable workflows → run "REFS QPF (Cape Canaveral)" once manually.
3. Settings → Pages → Deploy from branch → `gh-pages` / root.
4. Settings → Actions → General → Workflow permissions → Read and write.

## Verification
Each run saves its 48-h LPMM and AVRG totals to `archive/fcst/` (about 30 KB, int16).
The first run after the window closes pulls MRMS MultiSensor 48H Pass2, scores it,
deletes the forecast file, and appends to `archive/scores.json`. NCEP keeps MRMS
about 25 h, so a missed window (for example a multi-run NOMADS outage) is dropped.

## Tuning (top of `refs_lpmm_cape.py`)
- `EXTENT` map box; `SITES` point list (first entry drives the pad timeline)
- `THRESHOLDS` 48-h paintball; `EAS_THRESHOLDS` 6-h probability loop (0.01–3″ available)
- `FFRI_ARI_YEARS` recurrence intervals to map (1, 2, 5, 10, 25, 50, 100 available)
- `VERIFY_THRESHOLDS` ETS / frequency-bias thresholds
- `N_RUNS`, `RUN_SPACING`, `ACCUM_HOURS` with (N_RUNS-1)*RUN_SPACING + ACCUM_HOURS <= 60

Sources: https://nomads.ncep.noaa.gov/pub/data/nccf/com/refs/prod/ (about 2 days kept),
https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_48H_Pass2/ (about 25 h kept).
