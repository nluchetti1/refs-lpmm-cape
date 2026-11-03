# Space Coast REFS QPF

48-h REFS QPF for Cape Canaveral / KSC: LPMM time-lagged trend, EAS probabilities,
LPMM vs AVRG, flash flood / recurrence interval, per-site timelines for every
launch complex plus KXMR, KTTS and KCOF, MRMS verification tagged by flow regime,
and event look-backs across REFS, HRRR, RRFS, GFS, ECMWF IFS/AIFS, GEFS and ECMWF ENS.

## Setup
1. Push everything here, including `.github/` (workflows and ISSUE_TEMPLATE).
2. Settings → Actions → General → Workflow permissions → Read and write.
3. Settings → General → Features → Issues on (needed for public requests).
4. Issues → Labels → New label `lookback-request` (optional; titles starting
   "Look-back:" also trigger the request workflow).
5. Actions → "REFS QPF (Cape Canaveral)" → Run workflow (creates the `data` branch).
6. Settings → Pages → Deploy from branch → `gh-pages` / root.

## Look-backs
- **Anyone:** the site's Look back tab → Request on GitHub. This opens a pre-filled
  issue; a workflow validates it, builds the case, comments the link and closes
  the issue. A free GitHub account is required. Limit 3 per person per day;
  repeated windows return the existing case.
- **You:** Actions → "Look back" → Run workflow with start/end (YYYYMMDDHH UTC).
- **Automatic:** 24 h ending 12Z after each 12Z cycle (last 14 kept).

Any whole-hour window of 3–72 h. Models are included only where their output steps
land on the window edges: ECMWF IFS/ENS 3-hourly, AIFS 6-hourly, the others hourly.
Windows that aren't 6-h aligned make GEFS read every 3-h file (about 2.5x slower).
Coarse models are scored against MRMS averaged over their own grid box.
Ensembles show member mean, 75th percentile and max (an upper envelope, not a forecast).

## Branches
- `main` code · `gh-pages` site · `data` persistent state (force-pushed, no history):
  `archive/` forecasts awaiting scoring, scores, regime log; `cases/`; `site_cache/`.

## Flow regime
Pressure-weighted 10 m–700 mb wind, 25-km mean around XMR. Light < 5 kt; mixed when
vector-mean / mean speed < 0.5; otherwise NE/SE/SW/NW. Tags: XMR sounding (IEM, best
effort), REFS 1-h analysis proxy, run forecast. Edit `LIGHT_KT`, `MIXED_CONSISTENCY`,
`REGIME_*`, `RAOB_*` in `refs_lpmm_cape.py`.
