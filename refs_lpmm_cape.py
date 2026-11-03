"""
REFS QPF — Cape Canaveral / Space Coast
=======================================

Builds a static GitHub Pages site from operational REFS ensemble products on
NOMADS, plus a running verification against MRMS.

Inputs (all .../refs/prod/refs.YYYYMMDD/CC/ensprod/refs.tCCz.<type>.fFF.conus.grib2)
  lpmm  1-h localized probability-matched mean QPF      (latest + 2 time-lagged runs)
  avrg  1-h PMM/mean blend QPF                           (latest run)
  eas   6-h ensemble-agreement-scale exceedance probs    (latest run, f06..f48)
  ffri  6-h P(>FFG) and P(>N-yr recurrence interval)      (latest run, f06..f48)
  MRMS  MultiSensor_QPE_48H_Pass2 (verification)          https://mrms.ncep.noaa.gov/2D/

Inventories confirmed 2026-11-03 against NOMADS listings and the NCO product
inventory pages. NCEP MRMS keeps ~25 h of files, so each run's 48-h forecast is
archived to ./archive and scored on the first run after its window closes.

Outputs (./site)
  trend_48h.png  thresholds_48h.png  loop_06h_NN.png  eas_<thr>_NN.png
  lpmm_vs_avrg.png  ffri_48h.png  pad_timeline.png  verify.png  verify_map.png
  meta.json  verify.json  index.html
"""
import matplotlib
matplotlib.use("Agg")

import glob
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pygrib
import requests
from matplotlib.lines import Line2D
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# 1. CONFIG
# =============================================================================
BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/refs/prod"
FILE_TMPL = "{base}/refs.{date}/{hh}/ensprod/refs.t{hh}z.{ptype}.f{fhr:02d}.conus.grib2"
MRMS_TMPL = ("https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_48H_Pass2/"
             "MRMS_MultiSensor_QPE_48H_Pass2_00.00_{t:%Y%m%d-%H}0000.grib2.gz")

OUT_DIR = "site"
ARCHIVE_DIR = "archive"
FCST_DIR = os.path.join(ARCHIVE_DIR, "fcst")
SCORES_FILE = os.path.join(ARCHIVE_DIR, "scores.json")
VERIFY_LATEST = os.path.join(ARCHIVE_DIR, "verify_latest.npz")
INDEX_SRC = "index.html"

EXTENT = [-83.0, -79.0, 26.5, 30.5]       # [W, E, S, N]
CROP_PAD = 0.5

ACCUM_HOURS = 48
N_RUNS = 3
RUN_SPACING = 6
MAX_FHR = 60
WIN = 6                                   # h, loop / EAS / FFRI window
THRESHOLDS = [1.0, 2.0, 3.0, 5.0]         # in, 48-h LPMM paintball
EAS_THRESHOLDS = [0.25, 0.5, 1.0, 2.0]    # in, 6-h EAS panels (file also has 0.01, 3)
FFRI_ARI_YEARS = [2, 10]                  # recurrence intervals to map
VERIFY_THRESHOLDS = [0.5, 1.0, 2.0]       # in, 48-h ETS / frequency bias

EXPECTED_LATENCY = timedelta(hours=3, minutes=15)
POLL_INTERVAL_S = 300
MAX_POLLS = 12
MRMS_LATENCY = timedelta(minutes=70)      # Pass2 posted ~57 min after the hour
MRMS_RETENTION = timedelta(hours=23)      # NCEP server keeps ~25 h

MAX_WORKERS = 4
NBHD_KM = 10.0
MM_TO_IN = 0.0393701

SITES = {
    "LC-39A":             (28.6082, -80.6041, "*"),
    "SLC-40":             (28.5619, -80.5772, "*"),
    "Skid Strip (KXMR)":  (28.4676, -80.5669, "^"),
    "SLF (KTTS)":         (28.6150, -80.6945, "^"),
    "Patrick (KCOF)":     (28.2349, -80.6101, "^"),
    "Melbourne (KMLB)":   (28.1028, -80.6453, "o"),
    "Orlando (KMCO)":     (28.4294, -81.3089, "o"),
    "Daytona (KDAB)":     (29.1799, -81.0581, "o"),
}
PRIMARY_SITE = "LC-39A"
MAP_LABELS = ["LC-39A", "Skid Strip (KXMR)", "Melbourne (KMLB)", "Orlando (KMCO)", "Daytona (KDAB)"]

# =============================================================================
# 2. STYLE
# =============================================================================
BG, LAND, OCEAN, LINE, TEXT, MUTED = "#0f1a24", "#1c2a36", "#101d28", "#5b7183", "#e6edf2", "#8ea3b3"

CLEVS = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
CMAP_COLORS = [
    (0.31, 0.81, 0.81), (0, 1, 1), (0, 0.87, 0.5), (0, 0.75, 0), (0.5, 0.87, 0),
    (1, 1, 0), (1, 0.62, 0), (1, 0, 0), (1, 0.12, 0.5), (0.94, 0.25, 1),
    (0.5, 0.12, 1), (0.25, 0.25, 1), (0.12, 0.12, 0.5), (0.35, 0.35, 0.35),
    (0.5, 0.5, 0.5), (0.87, 0.87, 0.87), (0.93, 0.83, 0.73),
]
CMAP = mcolors.ListedColormap(CMAP_COLORS, "precip")
CMAP.set_over((0.85, 0.65, 0.47))
NORM = mcolors.BoundaryNorm(CLEVS, CMAP.N)

PLEVS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PCMAP = mcolors.ListedColormap(plt.get_cmap("plasma")(np.linspace(0.12, 1.0, len(PLEVS) - 1)), "prob")
PNORM = mcolors.BoundaryNorm(PLEVS, PCMAP.N)

DLEVS = [-3, -2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2, 3]
DCMAP = plt.get_cmap("BrBG", len(DLEVS) + 1)
DNORM = mcolors.BoundaryNorm(DLEVS, DCMAP.N, extend="both")

RUN_COLORS = ["#4fd1e8", "#f0b429", "#e0607e"]
RUN_LABELS = ["Latest", "−6 h", "−12 h"]
AVRG_COLOR = "#b59cf5"
EAS_COLORS = {0.25: "#9fb3c2", 0.5: "#4fd1e8", 1.0: "#f0b429", 2.0: "#e0607e"}

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": LINE,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "font.family": "DejaVu Sans",
})

# =============================================================================
# 3. NETWORK
# =============================================================================
def make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS + 2))
    s.headers["User-Agent"] = "refs-lpmm-cape/2.0 (GitHub Actions)"
    return s


SESSION = make_session()


def url_for(init, ptype, fhr):
    return FILE_TMPL.format(base=BASE_URL, date=init.strftime("%Y%m%d"),
                            hh=init.strftime("%H"), ptype=ptype, fhr=fhr)


def exists(url):
    try:
        return SESSION.head(url, timeout=20).status_code == 200
    except requests.RequestException:
        return False


def download(url, suffix=".grib2", gunzip=False):
    r = SESSION.get(url, timeout=90)
    if r.status_code != 200:
        raise FileNotFoundError(f"{r.status_code} {url}")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(gzip.decompress(r.content) if gunzip else r.content)
    return path


def cycle_floor(t):
    return t.replace(hour=(t.hour // RUN_SPACING) * RUN_SPACING, minute=0, second=0, microsecond=0)


def find_latest_cycle():
    for attempt in range(1, MAX_POLLS + 1):
        now = datetime.now(timezone.utc)
        expected = cycle_floor(now - EXPECTED_LATENCY)
        for c in [cycle_floor(now) - timedelta(hours=RUN_SPACING * k) for k in range(5)]:
            if exists(url_for(c, "lpmm", ACCUM_HOURS)):
                if c >= expected or attempt == MAX_POLLS:
                    return c
                break
        print(f"[wait] expected {expected:%Y%m%d %HZ} incomplete (attempt {attempt}/{MAX_POLLS})")
        time.sleep(POLL_INTERVAL_S)
    return None

# =============================================================================
# 4. GRIB
# =============================================================================
_GRID = {}
_LOGGED = set()


def _step_range(msg):
    try:
        return int(msg.startStep), int(msg.endStep)
    except Exception:
        sr = str(getattr(msg, "stepRange", ""))
        if "-" in sr:
            a, b = sr.split("-")
            return int(a), int(b)
        return None, int(sr) if sr.isdigit() else None


def _lower_limit(msg):
    try:
        return float(msg["lowerLimit"])
    except Exception:
        return float(msg["scaledValueOfLowerLimit"]) * 10.0 ** (-float(msg["scaleFactorOfLowerLimit"]))


def _is_apcp(m):
    return getattr(m, "discipline", 0) == 0 and m.parameterCategory == 1 and m.parameterNumber == 8


def _log_inventory(tag, msgs):
    key = tag.split()[0]          # once per product type
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    parts = []
    for m in msgs:
        try:
            ll = f" >{_lower_limit(m):g}"
        except Exception:
            ll = ""
        parts.append(f"#{m.messagenumber} {getattr(m, 'shortName', '?')} {_step_range(m)}{ll}")
    print(f"[inventory {tag}] " + "; ".join(parts))


def _set_grid(msg):
    lats, lons = msg.latlons()
    lons = np.where(lons > 180, lons - 360, lons)
    w, e, s, n = EXTENT
    inside = ((lats >= s - CROP_PAD) & (lats <= n + CROP_PAD) &
              (lons >= w - CROP_PAD) & (lons <= e + CROP_PAD))
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    rs, cs = slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)
    _GRID.update(full=lats.shape, rows=rs, cols=cs, lats=lats[rs, cs], lons=lons[rs, cs])
    print(f"[grid] full {lats.shape} -> cropped {_GRID['lats'].shape}")


def _crop(msg):
    if not _GRID:
        _set_grid(msg)
    vals = np.ma.filled(msg.values, 0.0)
    if vals.shape != _GRID["full"]:
        raise ValueError(f"grid {vals.shape} != {_GRID['full']}")
    return vals[_GRID["rows"], _GRID["cols"]].astype(np.float32)


def _pct(a):
    """Probabilities may be encoded as 0-1 or 0-100; return percent."""
    return a * 100.0 if 0 < np.nanmax(a) <= 1.0001 else a


def read_1h_qpf(path, fhr, tag):
    """1-h APCP (in) ending at fhr. f03/f06/... also hold 3-h/6-h sums, so match on step range."""
    with pygrib.open(path) as g:
        msgs = [m for m in g if _is_apcp(m)]
        if len(msgs) > 1:
            _log_inventory(f"{tag} f{fhr:02d}", msgs)
        pick = next((m for m in msgs if _step_range(m) == (fhr - 1, fhr)), None)
        if pick is None and len(msgs) == 1:
            pick = msgs[0]
        if pick is None:
            raise ValueError(f"no 1-h APCP at f{fhr:02d}: {[_step_range(m) for m in msgs]}")
        return _crop(pick) * MM_TO_IN


def read_eas_6h(path, fhr):
    """{threshold_in: P(%)} for the 6-h window ending at fhr."""
    out = {}
    with pygrib.open(path) as g:
        msgs = [m for m in g if _is_apcp(m)]
        _log_inventory(f"eas f{fhr:02d}", msgs)
        for m in msgs:
            if _step_range(m) != (fhr - WIN, fhr):
                continue
            thr = round(_lower_limit(m) * MM_TO_IN, 2)
            out[thr] = _pct(_crop(m))
    return out


def read_ffri_6h(path, fhr):
    """{'ffg': P(%), 'ari': {years: P(%)}} for the 6-h window ending at fhr."""
    out = {"ffg": None, "ari": {}}
    with pygrib.open(path) as g:
        msgs = list(g)
        _log_inventory(f"ffri f{fhr:02d}", msgs)
        for m in msgs:
            if _step_range(m) != (fhr - WIN, fhr):
                continue
            if _is_apcp(m):
                out["ari"][int(round(_lower_limit(m)))] = _pct(_crop(m))
            else:
                out["ffg"] = _pct(_crop(m))
    return out


def fetch_decoded(init, ptype, fhr, reader):
    path = download(url_for(init, ptype, fhr))
    try:
        return reader(path, fhr)
    finally:
        os.remove(path)


def fetch_many(init, ptype, fhrs, reader):
    """Decode the first file serially (sets the crop grid), the rest in parallel."""
    out = {}
    first, rest = fhrs[0], fhrs[1:]
    try:
        out[first] = fetch_decoded(init, ptype, first, reader)
    except Exception as e:
        print(f"  ! {ptype} f{first:02d}: {e}")
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_decoded, init, ptype, f, reader): f for f in rest}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                out[f] = fut.result()
            except Exception as e:
                print(f"  ! {ptype} f{f:02d}: {e}")
    return out


def fetch_qpf_run(init, ptype, lag):
    fhrs = list(range(lag + 1, lag + ACCUM_HOURS + 1))
    got = fetch_many(init, ptype, fhrs, lambda p, f: read_1h_qpf(p, f, ptype))
    if len(got) != len(fhrs):
        print(f"  skip {ptype} {init:%Y%m%d %HZ}: {len(got)}/{len(fhrs)} hours")
        return None
    return np.stack([got[f] for f in fhrs])


def window_hours():
    return list(range(WIN, ACCUM_HOURS + 1, WIN))

# =============================================================================
# 5. POINTS
# =============================================================================
SITE_IDX = {}


def _dist_km(lat, lon):
    lats, lons = _GRID["lats"], _GRID["lons"]
    dlat, dlon = np.radians(lats - lat), np.radians(lons - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def site_index(name):
    if name not in SITE_IDX:
        lat, lon, _ = SITES[name]
        d = _dist_km(lat, lon)
        SITE_IDX[name] = (np.unravel_index(np.argmin(d), d.shape), d <= NBHD_KM)
    return SITE_IDX[name]


def in_view():
    w, e, s, n = EXTENT
    la, lo = _GRID["lats"], _GRID["lons"]
    return (la >= s) & (la <= n) & (lo >= w) & (lo <= e)


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)

# =============================================================================
# 6. PLOTTING HELPERS
# =============================================================================
def add_basemap(ax, labels=True):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=OCEAN, edgecolor=LINE, linewidth=0.4, zorder=3)
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale("500k"), edgecolor="#33495a", linewidth=0.35, zorder=3)
    except Exception as e:
        print(f"[warn] counties unavailable: {e}")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#9fb3c2", linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor=LINE, linewidth=0.6, zorder=4)
    for name, (lat, lon, mk) in SITES.items():
        star = mk == "*"
        ax.plot(lon, lat, marker=mk, ms=7 if star else 4, mfc="#ffffff" if star else TEXT,
                mec=BG, mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
        if labels and name in MAP_LABELS:
            ax.text(lon + 0.05, lat + 0.04, name.split(" (")[0], fontsize=7, color=TEXT,
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.6))
    ax.spines["geo"].set_edgecolor(LINE)


def shade(ax, field, kind="qpf"):
    if kind == "qpf":
        data, lev, cm, nm, ext = np.ma.masked_less(field, CLEVS[0]), CLEVS, CMAP, NORM, "max"
    elif kind == "prob":
        data, lev, cm, nm, ext = np.ma.masked_less(field, PLEVS[0]), PLEVS, PCMAP, PNORM, "neither"
    else:
        data, lev, cm, nm, ext = np.ma.masked_inside(field, -0.1, 0.1), DLEVS, DCMAP, DNORM, "both"
    return ax.contourf(_GRID["lons"], _GRID["lats"], data, lev, cmap=cm, norm=nm, extend=ext,
                       alpha=0.92, transform=ccrs.PlateCarree(), zorder=2)


def colorbar(fig, cs, rect, label, ticks):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8)
    cb.set_label(label, fontsize=10, color=TEXT)
    cb.outline.set_edgecolor(LINE)


def annotate_max(ax, field, unit='"', fmt="{:.2f}", floor=0.1):
    masked = np.where(in_view(), field, -np.inf)
    j, i = np.unravel_index(np.argmax(masked), masked.shape)
    if field[j, i] >= floor:
        lat, lon = _GRID["lats"][j, i], _GRID["lons"][j, i]
        ax.plot(lon, lat, "x", color="#ffffff", ms=6, mew=1.4, transform=ccrs.PlateCarree(), zorder=7)
        ax.text(lon, lat - 0.12, fmt.format(field[j, i]) + unit, fontsize=8, ha="center", va="top",
                color="#ffffff", transform=ccrs.PlateCarree(), zorder=7)


def savefig(fig, name, dpi):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=dpi)
    plt.close(fig)


def fmt_init(t):
    return t.strftime("%a %d %b %HZ")

# =============================================================================
# 7. PRODUCTS
# =============================================================================
def plot_trend(runs, valid_txt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ax in enumerate(axes):
        add_basemap(ax, labels=(k == 0))
        if k < len(runs):
            tot = runs[k]["data"].sum(axis=0)
            cs = shade(ax, tot)
            annotate_max(ax, tot)
            ax.set_title(f"{RUN_LABELS[k]}  ·  {fmt_init(runs[k]['init'])} init", fontsize=11,
                         color=RUN_COLORS[k], fontweight="bold", pad=6)
        else:
            ax.set_title(f"{RUN_LABELS[k]}  ·  unavailable", fontsize=11, color=MUTED, pad=6)
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Precipitation (in)", CLEVS)
    fig.suptitle(f"REFS LPMM 48-h QPF, run-to-run trend\n{valid_txt}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "trend_48h.png", 170)


def plot_thresholds(runs, valid_txt):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.9, wspace=0.05, hspace=0.1)
    handles = [Line2D([0], [0], marker="s", ls="", ms=10, mfc=RUN_COLORS[k], mec="none",
                      label=f"{RUN_LABELS[k]} ({runs[k]['init']:%d/%HZ})") for k in range(len(runs))]
    for ax, thr in zip(axs.flat, THRESHOLDS):
        add_basemap(ax, labels=False)
        for k in reversed(range(len(runs))):
            tot = runs[k]["data"].sum(axis=0)
            if (tot >= thr).any():
                ax.contourf(_GRID["lons"], _GRID["lats"], tot, levels=[thr, 1e6], colors=[RUN_COLORS[k]],
                            alpha=0.55, transform=ccrs.PlateCarree(), zorder=2)
                ax.contour(_GRID["lons"], _GRID["lats"], tot, levels=[thr], colors=[RUN_COLORS[k]],
                           linewidths=1.1, transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(f"48-h LPMM ≥ {thr:g}\"", fontsize=12, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"REFS LPMM threshold paintball\n{valid_txt}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "thresholds_48h.png", 150)


def _frame(field, kind, title, subtitle, fname, cbar_label, ticks, unit, fmt, floor):
    fig = plt.figure(figsize=(8.4, 9))
    ax = fig.add_axes([0.03, 0.15, 0.94, 0.75], projection=ccrs.PlateCarree())
    add_basemap(ax)
    cs = shade(ax, field, kind)
    annotate_max(ax, field, unit=unit, fmt=fmt, floor=floor)
    fig.text(0.03, 0.955, title, fontsize=14, fontweight="bold")
    fig.text(0.03, 0.925, subtitle, fontsize=10, color=MUTED)
    colorbar(fig, cs, [0.08, 0.07, 0.84, 0.025], cbar_label, ticks)
    savefig(fig, fname, 115)


def plot_loop(run):
    frames, init = [], run["init"]
    for n, end in enumerate(window_hours(), start=1):
        start = end - WIN
        v0, v1 = init + timedelta(hours=start), init + timedelta(hours=end)
        fname = f"loop_06h_{n:02d}.png"
        _frame(run["data"][start:end].sum(axis=0), "qpf",
               f"REFS LPMM {WIN}-h QPF  ·  f{start:02d}–f{end:02d}",
               f"Valid {v0:%a %HZ} – {v1:%a %HZ %d %b}  ·  {init:%HZ %d %b} init",
               fname, "Precipitation (in)", CLEVS, '"', "{:.2f}", 0.1)
        frames.append({"file": fname, "start": v0.isoformat(), "end": v1.isoformat(),
                       "fhr": f"f{start:02d}-f{end:02d}"})
    return frames


def plot_eas_loop(init, eas):
    """eas: {fhr: {thr: P}} -> frames keyed by threshold."""
    out = {}
    for thr in EAS_THRESHOLDS:
        frames = []
        for n, end in enumerate(window_hours(), start=1):
            field = eas.get(end, {}).get(thr)
            if field is None:
                continue
            start = end - WIN
            v0, v1 = init + timedelta(hours=start), init + timedelta(hours=end)
            fname = f"eas_{str(thr).replace('.', 'p')}_{n:02d}.png"
            _frame(field, "prob",
                   f"REFS EAS  P(6-h QPF ≥ {thr:g}\")  ·  f{start:02d}–f{end:02d}",
                   f"Valid {v0:%a %HZ} – {v1:%a %HZ %d %b}  ·  {init:%HZ %d %b} init",
                   fname, "Probability (%)", PLEVS, "%", "{:.0f}", 5)
            frames.append({"file": fname, "start": v0.isoformat(), "end": v1.isoformat(),
                           "fhr": f"f{start:02d}-f{end:02d}"})
        if frames:
            out[f"{thr:g}"] = frames
    return out


def plot_lpmm_vs_avrg(lpmm_tot, avrg_tot, init, valid_txt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    add_basemap(axes[0]); add_basemap(axes[1], labels=False); add_basemap(axes[2], labels=False)
    cs = shade(axes[0], lpmm_tot); annotate_max(axes[0], lpmm_tot)
    shade(axes[1], avrg_tot); annotate_max(axes[1], avrg_tot)
    ds = shade(axes[2], lpmm_tot - avrg_tot, "diff")
    axes[0].set_title("LPMM", fontsize=12, fontweight="bold", color=RUN_COLORS[0])
    axes[1].set_title("AVRG (PMM / mean blend)", fontsize=12, fontweight="bold", color=AVRG_COLOR)
    axes[2].set_title("LPMM − AVRG", fontsize=12, fontweight="bold")
    colorbar(fig, cs, [0.04, 0.1, 0.6, 0.03], "Precipitation (in)", CLEVS)
    colorbar(fig, ds, [0.69, 0.1, 0.27, 0.03], "Difference (in), green = LPMM wetter", DLEVS)
    fig.suptitle(f"REFS 48-h QPF, {fmt_init(init)} init: LPMM vs AVRG\n{valid_txt}",
                 fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "lpmm_vs_avrg.png", 170)
    return True


def plot_ffri(ffri_max, init, valid_txt):
    panels = [("ffg", "6-h QPF > flash flood guidance")] + \
             [(y, f"6-h QPF > {y}-yr recurrence") for y in FFRI_ARI_YEARS]
    panels = [(k, t) for k, t in panels if ffri_max.get(k) is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 6.6),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ((key, title), ax) in enumerate(zip(panels, axes[0])):
        add_basemap(ax, labels=(k == 0))
        cs = shade(ax, ffri_max[key], "prob")
        annotate_max(ax, ffri_max[key], unit="%", fmt="{:.0f}", floor=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Highest neighborhood probability in any 6-h window (%)", PLEVS)
    fig.suptitle(f"REFS flash flood and recurrence interval, {fmt_init(init)} init\n{valid_txt}",
                 fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "ffri_48h.png", 150)
    return True


def plot_pad_timeline(runs, avrg, eas, valid_start):
    (j, i), _ = site_index(PRIMARY_SITE)
    times = [valid_start + timedelta(hours=h) for h in range(1, ACCUM_HOURS + 1)]
    has_eas = bool(eas)
    fig, axs = plt.subplots(3 if has_eas else 2, 1, figsize=(12, 9.5 if has_eas else 7), sharex=True,
                            gridspec_kw={"height_ratios": [1.1, 1, 0.9] if has_eas else [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.08, hspace=0.12)
    a1, a2 = axs[0], axs[1]
    for k in reversed(range(len(runs))):
        s = runs[k]["data"][:, j, i]
        lab = f"{RUN_LABELS[k]} LPMM ({runs[k]['init']:%d/%HZ})"
        if k == 0:
            a1.bar([t - timedelta(minutes=30) for t in times], s, width=timedelta(minutes=50),
                   color=RUN_COLORS[0], label=lab, zorder=3)
        else:
            a1.step(times, s, where="pre", color=RUN_COLORS[k], lw=1.4, label=lab, zorder=4)
        a2.plot(times, np.cumsum(s), color=RUN_COLORS[k], lw=2.2 if k == 0 else 1.5, label=lab)
    if avrg is not None:
        a1.step(times, avrg[:, j, i], where="pre", color=AVRG_COLOR, lw=1.2, ls="--", label="Latest AVRG", zorder=5)
        a2.plot(times, np.cumsum(avrg[:, j, i]), color=AVRG_COLOR, lw=1.5, ls="--", label="Latest AVRG")
    if has_eas:
        a3 = axs[2]
        edges = [valid_start + timedelta(hours=h - WIN) for h in window_hours()] + [valid_start + timedelta(hours=ACCUM_HOURS)]
        for thr in EAS_THRESHOLDS:
            vals = [eas.get(h, {}).get(thr) for h in window_hours()]
            if all(v is None for v in vals):
                continue
            y = [np.nan if v is None else v[j, i] for v in vals]
            a3.step(edges, y + [y[-1]], where="post", color=EAS_COLORS[thr], lw=1.8, label=f"≥ {thr:g}\"")
        a3.set_ylabel("EAS 6-h P (%)")
        a3.set_ylim(0, 100)
        a3.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)
    for a in axs:
        a.grid(color="#263848", lw=0.6)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color(LINE)
        d = valid_start.replace(hour=0) - timedelta(days=1)
        while d < times[-1]:
            a.axvspan(d - timedelta(hours=1), d + timedelta(hours=11), color="#070d12", alpha=0.85, zorder=0)
            d += timedelta(days=1)
        a.set_xlim(valid_start, times[-1])
    a1.set_ylabel("1-h QPF (in)")
    a2.set_ylabel("Accumulated (in)")
    a1.set_ylim(0, max(0.1, a1.get_ylim()[1]))
    a2.set_ylim(0, max(0.1, a2.get_ylim()[1]))
    h, l = a1.get_legend_handles_labels()
    a1.legend(h[::-1], l[::-1], loc="upper left", frameon=False, fontsize=9, ncol=4)
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
    lat, lon, _ = SITES[PRIMARY_SITE]
    fig.suptitle(f"REFS at {PRIMARY_SITE} ({lat:.3f}, {lon:.3f}), nearest grid point\nShaded = local night",
                 fontsize=13, fontweight="bold", y=0.975)
    savefig(fig, "pad_timeline.png", 150)

# =============================================================================
# 8. VERIFICATION (MRMS 48-h Pass2)
# =============================================================================
def to_i16(x):
    return np.clip(np.round(x * 100), 0, 32767).astype(np.int16)


def from_i16(x):
    return x.astype(np.float32) / 100.0


def archive_forecast(init, lpmm_tot, avrg_tot):
    os.makedirs(FCST_DIR, exist_ok=True)
    fn = os.path.join(FCST_DIR, f"refs_{init:%Y%m%d%H}.npz")
    if not os.path.exists(fn):
        np.savez_compressed(fn, lpmm=to_i16(lpmm_tot),
                            avrg=to_i16(avrg_tot) if avrg_tot is not None else np.zeros(0, np.int16))
        print(f"[archive] saved {fn}")


def fetch_mrms_48h(valid_end):
    """MRMS 48-h Pass2 QPE (in) averaged over a 3x3 (~3 km) box at each REFS point; NaN where missing."""
    url = MRMS_TMPL.format(t=valid_end)
    try:
        path = download(url, gunzip=True)
    except Exception as e:
        print(f"[mrms] {e}")
        return None
    try:
        with pygrib.open(path) as g:
            m = g.message(1)
            vals = np.ma.filled(m.values, -999.0).astype(np.float32)
            lat0 = float(m["latitudeOfFirstGridPointInDegrees"])
            lon0 = float(m["longitudeOfFirstGridPointInDegrees"])
            dj = float(m["jDirectionIncrementInDegrees"])
            di = float(m["iDirectionIncrementInDegrees"])
            jpos = int(m["jScansPositively"])
    finally:
        os.remove(path)
    lon0 = lon0 - 360 if lon0 > 180 else lon0
    lats, lons = _GRID["lats"], _GRID["lons"]
    rows = np.rint(((lats - lat0) if jpos else (lat0 - lats)) / dj).astype(int)
    cols = np.rint((lons - lon0) / di).astype(int)
    stack = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = np.clip(rows + dr, 0, vals.shape[0] - 1)
            c = np.clip(cols + dc, 0, vals.shape[1] - 1)
            v = vals[r, c]
            stack.append(np.where(v >= 0, v, np.nan))
    with np.errstate(invalid="ignore"):
        obs = np.nanmean(np.stack(stack), axis=0) * MM_TO_IN
    print(f"[mrms] {valid_end:%Y%m%d %HZ}: {np.isfinite(obs).mean():.0%} valid, max {np.nanmax(obs):.2f}\"")
    return obs


def categorical(f, o, thr):
    fy, oy = f >= thr, o >= thr
    hits, fa, miss = int((fy & oy).sum()), int((fy & ~oy).sum()), int((~fy & oy).sum())
    n = f.size
    hr = (hits + fa) * (hits + miss) / n if n else 0
    den = hits + fa + miss - hr
    ets = (hits - hr) / den if den > 0 else None
    fbias = (hits + fa) / (hits + miss) if (hits + miss) > 0 else None
    return r2(ets) if ets is not None else None, r2(fbias) if fbias is not None else None


def score(init, lpmm, avrg, obs):
    mask = in_view() & np.isfinite(obs)
    o = obs[mask]
    (j, i), _ = site_index(PRIMARY_SITE)
    rec = {"init": init.isoformat(), "valid_end": (init + timedelta(hours=ACCUM_HOURS)).isoformat(),
           "n": int(mask.sum()), "obs_mean": r2(o.mean()), "obs_max": r2(o.max()),
           "pad": {"obs": r2(obs[j, i]), "lpmm": r2(lpmm[j, i]), "avrg": r2(avrg[j, i]) if avrg is not None else None}}
    for name, fc in (("lpmm", lpmm), ("avrg", avrg)):
        if fc is None:
            continue
        f = fc[mask]
        e = {f"{t:g}": categorical(f, o, t) for t in VERIFY_THRESHOLDS}
        rec[name] = {"mean": r2(f.mean()),
                     "bias_ratio": r2(f.sum() / o.sum()) if o.sum() > 0.01 * o.size else None,
                     "ets": {k: v[0] for k, v in e.items()},
                     "fbias": {k: v[1] for k, v in e.items()}}
    return rec


def verify_pending():
    scores = json.load(open(SCORES_FILE)) if os.path.exists(SCORES_FILE) else []
    done = {s["init"] for s in scores}
    now = datetime.now(timezone.utc)
    newest = None
    for fn in sorted(glob.glob(os.path.join(FCST_DIR, "refs_*.npz"))):
        init = datetime.strptime(os.path.basename(fn)[5:15], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        vend = init + timedelta(hours=ACCUM_HOURS)
        if init.isoformat() in done:
            os.remove(fn); continue
        if vend > now - MRMS_LATENCY:
            continue
        z = np.load(fn)
        if vend < now - MRMS_RETENTION or z["lpmm"].shape != _GRID["lats"].shape:
            print(f"[verify] dropping {fn} (MRMS no longer on server or grid changed)")
            os.remove(fn); continue
        obs = fetch_mrms_48h(vend)
        if obs is None:
            continue
        lp = from_i16(z["lpmm"])
        av = from_i16(z["avrg"]) if z["avrg"].size else None
        scores.append(score(init, lp, av, obs))
        newest = (init, lp, av, obs)
        os.remove(fn)
        print(f"[verify] scored {init:%Y%m%d %HZ}")
    scores.sort(key=lambda s: s["init"])
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=0)
    if newest:
        init, lp, av, obs = newest
        np.savez_compressed(VERIFY_LATEST, init=init.isoformat(), lpmm=to_i16(lp),
                            avrg=to_i16(av) if av is not None else np.zeros(0, np.int16),
                            obs=np.where(np.isfinite(obs), to_i16(np.nan_to_num(obs)), -1).astype(np.int16))
    return scores


def plot_verify_map():
    if not os.path.exists(VERIFY_LATEST):
        return False
    z = np.load(VERIFY_LATEST)
    if z["lpmm"].shape != _GRID["lats"].shape:
        return False
    init = datetime.fromisoformat(str(z["init"]))
    obs = np.where(z["obs"] >= 0, from_i16(z["obs"]), np.nan)
    fields = [("MRMS 48-h QPE (Pass 2)", obs, TEXT), ("LPMM", from_i16(z["lpmm"]), RUN_COLORS[0])]
    if z["avrg"].size:
        fields.append(("AVRG", from_i16(z["avrg"]), AVRG_COLOR))
    fig, axes = plt.subplots(1, len(fields), figsize=(5.4 * len(fields), 6.6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ((title, fld, col), ax) in enumerate(zip(fields, axes)):
        add_basemap(ax, labels=(k == 0))
        cs = shade(ax, np.nan_to_num(fld))
        annotate_max(ax, np.nan_to_num(fld))
        ax.set_title(title, fontsize=12, fontweight="bold", color=col)
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Precipitation (in)", CLEVS)
    vend = init + timedelta(hours=ACCUM_HOURS)
    fig.suptitle(f"Most recent verified run: {fmt_init(init)} init\n"
                 f"Valid {init:%HZ %a %d %b} to {vend:%HZ %a %d %b}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "verify_map.png", 150)
    return True


def plot_verify_scores(scores):
    if not scores:
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.22)
    for key, col, lab in (("lpmm", RUN_COLORS[0], "LPMM"), ("avrg", AVRG_COLOR, "AVRG")):
        pts = [(s["pad"]["obs"], s["pad"][key]) for s in scores
               if s["pad"].get(key) is not None and s["pad"]["obs"] is not None]
        if pts:
            o, f = zip(*pts)
            a1.scatter(o, f, s=22, color=col, label=lab, alpha=0.85, edgecolor="none")
        series = [(datetime.fromisoformat(s["init"]), s.get(key, {}).get("ets", {}).get("1"))
                  for s in scores[-120:] if key in s]
        series = [(t, v) for t, v in series if v is not None]
        if series:
            t, v = zip(*series)
            a2.plot(t, v, "-o", ms=3, lw=1.2, color=col, label=f"{lab} (mean {np.mean(v):.2f})")
    lim = max(0.5, a1.get_xlim()[1], a1.get_ylim()[1])
    a1.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    a1.set_xlim(0, lim); a1.set_ylim(0, lim)
    a1.set_xlabel("MRMS 48-h (in)"); a1.set_ylabel("REFS 48-h (in)")
    a1.set_title(f"{PRIMARY_SITE}, {len(scores)} runs", fontsize=11, fontweight="bold")
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_ylabel("ETS, 48-h ≥ 1\"")
    a2.set_title("Domain equitable threat score by run (blank = no 1\" observed or forecast)",
                 fontsize=11, fontweight="bold")
    loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
    a2.xaxis.set_major_locator(loc)
    a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    t_all = [datetime.fromisoformat(s["init"]) for s in scores[-120:]]
    pad = timedelta(days=1)
    a2.set_xlim(min(t_all) - pad, max(t_all) + pad)
    for a in (a1, a2):
        a.grid(color="#263848", lw=0.6)
        a.legend(frameon=False, fontsize=9, loc="upper left")
        for sp in a.spines.values():
            sp.set_color(LINE)
    fig.suptitle("REFS vs MRMS 48-h QPE", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "verify.png", 140)
    return True

# =============================================================================
# 9. META
# =============================================================================
def build_meta(runs, avrg, eas, ffri_max, valid_start, valid_end, frames, eas_frames, products, scores):
    sites = []
    for name, (lat, lon, mk) in SITES.items():
        (j, i), nb = site_index(name)
        per_run = []
        for k, run in enumerate(runs):
            s = run["data"][:, j, i]
            h = int(np.argmax(s))
            per_run.append({"run": RUN_LABELS[k], "total": r2(s.sum()),
                            "nbhd_max": r2(run["data"].sum(axis=0)[nb].max()) if nb.any() else None,
                            "max_1h": r2(s[h]),
                            "max_1h_valid": (valid_start + timedelta(hours=h + 1)).isoformat()})
        eas_max = {}
        for thr in EAS_THRESHOLDS:
            v = [eas[h][thr][j, i] for h in window_hours() if thr in eas.get(h, {})]
            if v:
                eas_max[f"{thr:g}"] = int(round(max(v)))
        sites.append({
            "name": name, "lat": lat, "lon": lon, "pad": mk == "*", "runs": per_run,
            "avrg_total": r2(avrg[:, j, i].sum()) if avrg is not None else None,
            "eas_max": eas_max,
            "ffg_max": int(round(ffri_max["ffg"][j, i])) if ffri_max.get("ffg") is not None else None,
        })

    tot = runs[0]["data"].sum(axis=0)
    masked = np.where(in_view(), tot, -1)
    j, i = np.unravel_index(np.argmax(masked), masked.shape)

    verify_summary = None
    if scores:
        def mean_of(key, path):
            v = [s.get(key, {}).get(path[0], {}).get(path[1]) for s in scores[-30:]]
            v = [x for x in v if x is not None]
            return (r2(np.mean(v)), len(v)) if v else (None, 0)
        verify_summary = {
            "n": len(scores),
            "ets1_lpmm": mean_of("lpmm", ("ets", "1")), "ets1_avrg": mean_of("avrg", ("ets", "1")),
            "fbias1_lpmm": mean_of("lpmm", ("fbias", "1")), "fbias1_avrg": mean_of("avrg", ("fbias", "1")),
            "recent": scores[-10:][::-1],
        }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": BASE_URL, "accum_hours": ACCUM_HOURS,
        "valid_start": valid_start.isoformat(), "valid_end": valid_end.isoformat(),
        "runs": [{"label": RUN_LABELS[k], "init": r["init"].isoformat(), "color": RUN_COLORS[k]}
                 for k, r in enumerate(runs)],
        "primary_site": PRIMARY_SITE, "nbhd_km": NBHD_KM,
        "thresholds": THRESHOLDS, "eas_thresholds": EAS_THRESHOLDS,
        "domain_max": {"value": r2(tot[j, i]), "lat": round(float(_GRID["lats"][j, i]), 3),
                       "lon": round(float(_GRID["lons"][j, i]), 3)},
        "sites": sites, "frames": frames, "eas_frames": eas_frames, "products": products,
        "verify": verify_summary,
        "colorscale": {"levels": CLEVS, "colors": [mcolors.to_hex(c) for c in CMAP_COLORS]},
    }

# =============================================================================
# 10. MAIN
# =============================================================================
def optional(label, fn, *a):
    """Run a non-core product; a failure is logged and the rest of the site still builds."""
    try:
        return fn(*a)
    except Exception:
        print(f"[warn] {label} failed:\n{traceback.format_exc()}")
        return None


def main():
    assert (N_RUNS - 1) * RUN_SPACING + ACCUM_HOURS <= MAX_FHR, "time-lag window exceeds f60"
    os.makedirs(OUT_DIR, exist_ok=True)

    latest = find_latest_cycle()
    if latest is None:
        print("No complete REFS cycle found; leaving the published site unchanged.")
        sys.exit(1)
    print(f"Latest complete cycle: {latest:%Y%m%d %HZ}")

    runs = []
    for k in range(N_RUNS):
        lag = k * RUN_SPACING
        init = latest - timedelta(hours=lag)
        print(f"LPMM {RUN_LABELS[k]}: {init:%Y%m%d %HZ} f{lag + 1:02d}-f{lag + ACCUM_HOURS:02d}")
        data = fetch_qpf_run(init, "lpmm", lag)
        if data is None:
            if k == 0:
                print("Latest run incomplete; aborting.")
                sys.exit(1)
            continue
        runs.append({"init": init, "data": data})

    print("AVRG latest")
    avrg = optional("avrg", fetch_qpf_run, latest, "avrg", 0)
    print("EAS latest")
    eas = optional("eas", fetch_many, latest, "eas", window_hours(), read_eas_6h) or {}
    print("FFRI latest")
    ffri = optional("ffri", fetch_many, latest, "ffri", window_hours(), read_ffri_6h) or {}

    ffri_max = {}
    if ffri:
        ffg = [v["ffg"] for v in ffri.values() if v["ffg"] is not None]
        ffri_max["ffg"] = np.max(np.stack(ffg), axis=0) if ffg else None
        for y in FFRI_ARI_YEARS:
            a = [v["ari"][y] for v in ffri.values() if y in v["ari"]]
            ffri_max[y] = np.max(np.stack(a), axis=0) if a else None

    valid_start, valid_end = latest, latest + timedelta(hours=ACCUM_HOURS)
    valid_txt = f"Valid {valid_start:%HZ %a %d %b} to {valid_end:%HZ %a %d %b %Y}"
    lpmm_tot = runs[0]["data"].sum(axis=0)
    avrg_tot = avrg.sum(axis=0) if avrg is not None else None

    # verification bookkeeping first, so a plotting failure can't lose the archive
    archive_forecast(latest, lpmm_tot, avrg_tot)
    scores = optional("verification", verify_pending) or []

    plot_trend(runs, valid_txt)
    plot_thresholds(runs, valid_txt)
    frames = plot_loop(runs[0])
    eas_frames = optional("eas loop", plot_eas_loop, latest, eas) or {}
    products = {
        "eas": bool(eas_frames),
        "avrg": avrg_tot is not None and bool(optional("lpmm vs avrg", plot_lpmm_vs_avrg,
                                                        lpmm_tot, avrg_tot, latest, valid_txt)),
        "ffri": any(v is not None for v in ffri_max.values())
                and bool(optional("ffri", plot_ffri, ffri_max, latest, valid_txt)),
        "verify_map": bool(optional("verify map", plot_verify_map)),
        "verify": bool(optional("verify scores", plot_verify_scores, scores)),
    }
    plot_pad_timeline(runs, avrg, eas, valid_start)

    meta = build_meta(runs, avrg, eas, ffri_max, valid_start, valid_end, frames, eas_frames, products, scores)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if os.path.exists(INDEX_SRC):
        shutil.copy(INDEX_SRC, os.path.join(OUT_DIR, "index.html"))
    open(os.path.join(OUT_DIR, ".nojekyll"), "w").close()

    p = next(s for s in meta["sites"] if s["name"] == PRIMARY_SITE)
    print(f"Done. {PRIMARY_SITE} 48-h LPMM {p['runs'][0]['total']}\", AVRG {p['avrg_total']}\"; "
          f"products {products}; verified runs {len(scores)}")


if __name__ == "__main__":
    main()
