# Space Coast REFS QPF

48-h REFS QPF for Cape Canaveral / KSC: LPMM time-lagged trend, EAS probabilities,
LPMM vs AVRG, flash flood / recurrence interval, pad timeline, MRMS verification
tagged by sea-breeze flow regime, and event look-backs comparing REFS, HRRR and RRFS.

## Setup
1. Push everything here (including `.github/`). Delete any old `archive/` folder on
   main; the first run copies it into the data branch if it still exists.
2. Settings → Actions → General → Workflow permissions → Read and write.
3. Actions → "REFS QPF (Cape Canaveral)" → Run workflow (creates the `data` branch).
4. Settings → Pages → Deploy from branch → `gh-pages` / root.

## Branches
- `main`   code only
- `gh-pages` published site (rebuilt every run)
- `data`   persistent state, force-pushed without history:
  `archive/` forecasts awaiting scoring, scores, regime log;
  `cases/` look-backs; `site_cache/` last site (lets the look-back workflow republish)

## Look back
Actions → "Look back" → Run workflow with `end` (YYYYMMDDHH UTC, 00/06/12/18),
`hours` (6/12/24/48/72) and a title. About 5 min. A 24-h case also builds
automatically after each 12Z cycle (last 14 kept; manual cases kept forever).
REFS archive on AWS starts 2026-08-13. Windows over 48 h are past REFS (f60) and
HRRR (f48) range once lead time is added, so 72-h cases show RRFS only.

## Flow regime
Pressure-weighted 10 m–700 mb mean wind, 25-km mean around XMR. Light < 5 kt;
mixed when vector-mean / mean speed < 0.5; otherwise NE/SE/SW/NW by quadrant.
Each scored run gets up to three tags: forecast (REFS mean f06–f48), analysis
proxy (REFS 1-h mean from each later cycle), and XMR sounding (IEM, best effort).
Verification groups by sounding, then analysis proxy, then forecast.
Edit `LIGHT_KT`, `MIXED_CONSISTENCY`, `REGIME_*` and `RAOB_*` in `refs_lpmm_cape.py`.

Sources: NOMADS refs/prod (~2 days) and AWS noaa-rrfs-ops-pds; MRMS NCEP (~25 h)
and AWS noaa-mrms-pds; HRRR AWS noaa-hrrr-bdp-pds; soundings mesonet.agron.iastate.edu.
