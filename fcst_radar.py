"""
Forecast "radar" loops for the situational awareness board: simulated composite reflectivity
coloured by precipitation type, with sea-level pressure isobars and H/L centres, over a regional
map wide enough to see systems coming.

  RRFS   3 km, hourly frames to 60-84 h   AWS noaa-rrfs-ops-pds, then NOMADS rrfs/prod
  GFS    0.25 deg, 3-hourly to 72 h       AWS noaa-gfs-bdp-pds, then NOMADS gfs/prod

Only the needed GRIB records are downloaded (byte ranges from each file's .idx). Frames are
written to site/radar/ and described in meta["radar"]; a run whose model cycle is unchanged
reuses the previous frames from data/site_cache.

NOTE: the .idx field strings below follow the usual NCEP inventories (REFC:entire atmosphere,
PRMSL/MSLMA:mean sea level, CRAIN/CSNOW/CFRZR/CICEP:surface). They could not be probed live
from the build sandbox; the first Actions log prints what each model actually matched.
"""
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pygrib
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

import refs_qpf as R

RRFS_URLS = ["https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.prslev.3km.f{f:03d}.conus.grib2",
             "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.prslev.3km.f{f:03d}.conus.grib2"]
GFS_URLS = ["https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p25.f{f:03d}",
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p25.f{f:03d}"]

MODELS = {
    # cycles: inits tried (6-hourly synoptic runs are the long ones); lengths: longest first
    "rrfs": {"label": "RRFS 3 km", "urls": RRFS_URLS, "step": 1, "lengths": [72, 60, 48, 36],
             "latency_h": 2, "smooth": 8.0, "note": "hourly"},
    "gfs": {"label": "GFS", "urls": GFS_URLS, "step": 3, "lengths": [72],
            "latency_h": 4, "smooth": 1.0, "note": "every 3 h"},
}
MODEL_ORDER = ["rrfs", "gfs"]
WORKERS = 6
OUT_SUB = "radar"

# idx line filter: composite reflectivity, MSLP, instantaneous categorical precip type
_KEEP = re.compile(r":(REFC:entire atmosphere|PRMSL:mean sea level|MSLMA:mean sea level|MSLET:mean sea level|"
                   r"(CRAIN|CSNOW|CFRZR|CICEP):surface:\d+ (hour|day) fcst)")

DBZ = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75]
RAIN = ["#a6e6a6", "#6fd171", "#3eb44c", "#1f8f35", "#11702a", "#f7e540", "#f5b52e",
        "#f07c24", "#e2401f", "#b81b1d", "#d13cd0", "#f4b3f2"]
SNOW = ["#d6ebff", "#b0d5fb", "#88bdf4", "#62a2ea", "#4487dc", "#2f6ccb",
        "#2156b4", "#18439a", "#123480", "#0d2766", "#0a1d4f", "#07143a"]
MIX = ["#ffd6ee", "#f9b2dd", "#f08dca", "#e468b5", "#d1489f", "#ba2d89",
       "#9f1b73", "#84105f", "#6a094c", "#52053b", "#3d032c", "#2a021e"]
NORM = mcolors.BoundaryNorm(DBZ, len(RAIN))
CMAPS = {k: mcolors.ListedColormap(v) for k, v in (("rain", RAIN), ("snow", SNOW), ("mix", MIX))}

EXTENT = None                    # board.radar_extent from site_config.json
CITIES = []                      # board.radar_cities: [[name, lat, lon], ...]
TZ = ZoneInfo("America/New_York")


def configure():
    cfg = R.load_json(R.CONFIG_FILE, {}) if os.path.exists(R.CONFIG_FILE) else {}
    b = cfg.get("board", {})
    global EXTENT, CITIES, TZ
    w, e, s, n = R.EXTENT
    EXTENT = b.get("radar_extent") or [w - 6, e + 6, s - 4, n + 4]
    CITIES = b.get("radar_cities", [])
    TZ = ZoneInfo(b.get("timezone", "America/New_York"))


# --------------------------------------------------------------------------- data
def _urls(model, init, f):
    return [u.format(d=init, f=f) for u in MODELS[model]["urls"]]


def find_init(model):
    """Newest 6-hourly cycle whose final frame is posted. Returns (init, length) or (None, None)."""
    m = MODELS[model]
    now = R.utcnow()
    base = (now - timedelta(hours=m["latency_h"])).replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=base.hour // 6 * 6)
    for k in range(5):
        init = base - timedelta(hours=6 * k)
        for length in m["lengths"]:
            if any(R.exists(u + ".idx") for u in _urls(model, init, length)):
                return init, length
    return None, None


def read_frame(model, init, f):
    path = R.download_subset(_urls(model, init, f), lambda l: bool(_KEEP.search(l)))
    out, crop = {}, None
    try:
        with pygrib.open(path) as g:
            for msg in g:
                sn = msg.shortName.lower()
                key = {"refc": "refc", "prmsl": "mslp", "mslma": "mslp", "mslet": "mslp",
                       "crain": "crain", "csnow": "csnow", "cfrzr": "cfrzr", "cicep": "cicep"}.get(sn)
                if key is None or (key == "mslp" and "mslp" in out):
                    continue
                if crop is None:
                    crop = R.make_crop(msg, EXTENT, pad=0.5)
                vals = np.ma.filled(msg.values.astype(np.float32), np.nan)
                out[key] = vals[crop["rows"], crop["cols"]]
    finally:
        os.remove(path)
    if "refc" not in out:
        raise ValueError(f"{model} f{f:03d}: no REFC record")
    return out, crop


# --------------------------------------------------------------------------- drawing
class RadarRenderer:
    """One figure and basemap reused for every frame; only the data artists are redrawn."""

    def __init__(self, model):
        w, e, s, n = EXTENT
        aspect = (n - s) / ((e - w) * np.cos(np.radians((n + s) / 2)))
        self.fig = plt.figure(figsize=(9.0, 9.0 * aspect * 0.94 + 0.9))
        self.ax = self.fig.add_axes([0.0, 0.08, 1.0, 0.84], projection=ccrs.PlateCarree())
        ax = self.ax
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0e1b26", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#22313d", zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#0e1b26", edgecolor="#4d6476", linewidth=0.4, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#a9bccb", linewidth=0.7, zorder=4)
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#6d8597", linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#6d8597", linewidth=0.6, zorder=4)
        pc = ccrs.PlateCarree()
        for c in CITIES:                                  # [name, lat, lon] or [name, lat, lon, "left"]
            name, la, lo = c[:3]
            left = len(c) > 3 and c[3] == "left"
            ax.plot(lo, la, marker="o", ms=3.5, mfc="#e6edf2", mec="#000000", mew=0.6, transform=pc, zorder=9)
            ax.text(lo + (-0.12 if left else 0.12), la + 0.08, name, fontsize=9, color="#ffffff", transform=pc,
                    zorder=9, ha="right" if left else "left", path_effects=_halo())
        plat, plon, _ = R.SITES[R.PRIMARY_SITE]
        ax.plot(plon, plat, marker="*", ms=15, mfc="#ffd84a", mec="#000000", mew=1.0, transform=pc, zorder=11)
        ax.text(plon + 0.2, plat, R.SITE_LABELS.get(R.PRIMARY_SITE, R.PRIMARY_SITE), fontsize=10, fontweight="bold",
                color="#ffd84a", va="center", transform=pc, zorder=11, path_effects=_halo(3))
        ax.spines["geo"].set_edgecolor("#4d6476")
        self.t1 = self.fig.text(0.015, 0.955, "", fontsize=15, fontweight="bold", color="#ffffff")
        self.t2 = self.fig.text(0.985, 0.955, "", fontsize=12, color="#c9d6e0", ha="right")
        self._legend()
        self.dyn = []
        self.smooth = MODELS[model]["smooth"]
        self.label = MODELS[model]["label"]

    def _legend(self):
        x0, y0, w, h = 0.06, 0.035, 0.24, 0.016
        for k, (name, cm) in enumerate((("Rain", CMAPS["rain"]), ("Snow", CMAPS["snow"]), ("Ice / mix", CMAPS["mix"]))):
            ax = self.fig.add_axes([x0 + k * 0.31, y0, w, h])
            ax.imshow(np.arange(cm.N)[None, :], cmap=cm, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            self.fig.text(x0 + k * 0.31, y0 + h + 0.006, name, fontsize=10, color="#e6edf2")
        self.fig.text(x0 + 2 * 0.31 + w, y0 + h + 0.006, "light → heavy", fontsize=9, color="#8ea3b3", ha="right")

    def draw(self, fields, crop, init, f, path):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        ax, pc = self.ax, ccrs.PlateCarree()
        lats, lons = crop["lats"], crop["lons"]
        z = np.where(np.isfinite(fields["refc"]), fields["refc"], -99)
        cls = np.zeros(z.shape, np.int8)                              # 0 rain (default), 1 snow, 2 mix
        if "csnow" in fields:
            cls[np.nan_to_num(fields["csnow"]) >= 1] = 1
        for k in ("cfrzr", "cicep"):
            if k in fields:
                cls[np.nan_to_num(fields[k]) >= 1] = 2
        for c, name in enumerate(("rain", "snow", "mix")):
            data = np.ma.masked_where((z < DBZ[0]) | (cls != c), z)
            if data.count():
                self.dyn.append(ax.pcolormesh(lons, lats, data, cmap=CMAPS[name], norm=NORM,
                                              transform=pc, zorder=2, shading="auto", rasterized=True))
        if "mslp" in fields:
            self.dyn += self._isobars(fields["mslp"], lats, lons)
        valid = init + timedelta(hours=f)
        loc = valid.astimezone(TZ)
        self.t1.set_text(f"{loc:%a %b %-d, %-I %p %Z}  ({valid:%HZ})")
        self.t2.set_text(f"{self.label} forecast radar · hour {f} of the {init:%HZ %a} run")
        self.fig.savefig(path, dpi=150, facecolor="#0f1a24")
        _quantize(path)

    def _isobars(self, p, lats, lons):
        ax, pc = self.ax, ccrs.PlateCarree()
        p = np.where(np.isfinite(p), p, np.nanmean(p))
        hpa = gaussian_filter(p / 100.0 if np.nanmax(p) > 2000 else p, self.smooth)
        levs = np.arange(960, 1052, 4)
        out = []
        cs = ax.contour(lons, lats, hpa, levels=levs, colors="#ffffff", linewidths=1.3, alpha=0.9,
                        transform=pc, zorder=5)
        ax.clabel(cs, fmt="%d", fontsize=9, inline=True, inline_spacing=2, colors="#ffffff")
        out.append(cs)                       # ContourSet.remove() also removes its labels
        # H / L at well-separated extrema inside the map
        size = max(5, int(round(min(hpa.shape) / 5)))
        for kind, filt, color in (("H", maximum_filter, "#4aa3ff"), ("L", minimum_filter, "#ff5a5a")):
            ext = (hpa == filt(hpa, size=size, mode="nearest"))
            edge = max(2, size // 3)
            ext[:edge, :] = ext[-edge:, :] = False
            ext[:, :edge] = ext[:, -edge:] = False
            for j, i in zip(*np.where(ext)):
                v = hpa[j, i]
                if kind == "H" and v < 1012 or kind == "L" and v > 1012:
                    continue
                out.append(ax.text(lons[j, i], lats[j, i], kind, fontsize=20, fontweight="bold", color=color,
                                   ha="center", va="center", transform=pc, zorder=7, path_effects=_halo(3)))
                out.append(ax.text(lons[j, i], lats[j, i], f"\n\n{v:.0f}", fontsize=8, color=color,
                                   ha="center", va="center", transform=pc, zorder=7, path_effects=_halo(2)))
        return out


def _halo(w=2.5):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground="#000000")]


def _quantize(path):
    """Palette PNG: about a third of the size, no visible loss on a radar map."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    except Exception as e:
        print(f"[radar] quantize skipped: {e}")


# --------------------------------------------------------------------------- build
def build_model(model, out_dir, previous=None):
    m = MODELS[model]
    init, length = find_init(model)
    if not init:
        raise FileNotFoundError(f"{model}: no recent cycle with a posted f{m['lengths'][-1]:03d}")
    hours = list(range(m["step"], length + 1, m["step"]))
    sub = os.path.join(out_dir, OUT_SUB)
    os.makedirs(sub, exist_ok=True)
    cache_sub = os.path.join(R.SITE_CACHE, OUT_SUB)
    if previous and previous.get("init") == init.isoformat() and len(previous.get("frames", [])) >= len(hours) * 0.9 \
            and all(os.path.exists(os.path.join(R.SITE_CACHE, fr["img"])) for fr in previous["frames"]):
        for fr in previous["frames"]:
            shutil.copy(os.path.join(R.SITE_CACHE, fr["img"]), os.path.join(out_dir, fr["img"]))
        print(f"[radar] {model} {init:%d/%HZ} unchanged; reused {len(previous['frames'])} frames")
        return previous
    print(f"[radar] {model} {init:%Y%m%d %HZ}: {len(hours)} frames f{hours[0]:03d}-f{hours[-1]:03d}")

    def fetch(f):
        try:
            return f, read_frame(model, init, f)
        except Exception as e:
            print(f"[radar] {model} f{f:03d}: {e}")
            return f, None

    renderer = RadarRenderer(model)
    frames, logged = [], False
    with ThreadPoolExecutor(WORKERS) as ex:
        for f, got in ex.map(fetch, hours):                  # results arrive in order; draw as they come
            if not got:
                continue
            fields, crop = got
            if not logged:
                print(f"[radar] {model} fields decoded: {sorted(fields)}; grid {crop['lats'].shape}")
                logged = True
            img = f"{OUT_SUB}/{model}_f{f:03d}.png"
            renderer.draw(fields, crop, init, f, os.path.join(out_dir, img))
            frames.append({"f": f, "valid": (init + timedelta(hours=f)).isoformat(), "img": img})
    plt.close(renderer.fig)
    if len(frames) < max(3, len(hours) // 2):
        raise ValueError(f"{model}: only {len(frames)} of {len(hours)} frames")
    return {"label": m["label"], "note": m["note"], "init": init.isoformat(), "frames": frames}


def build_radar(out_dir, previous=None):
    """meta['radar'] block: {'models': {key: {...}}, 'order': [...], 'built': iso}."""
    configure()
    previous = (previous or {}).get("models", {})
    models = {}
    for key in MODEL_ORDER:
        blk = R.optional(f"radar {key}", build_model, key, out_dir, previous.get(key))
        if not blk and previous.get(key):                 # keep the last good loop rather than none
            blk = previous[key]
            os.makedirs(os.path.join(out_dir, OUT_SUB), exist_ok=True)
            for fr in blk.get("frames", []):
                src = os.path.join(R.SITE_CACHE, fr["img"])
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(out_dir, fr["img"]))
            print(f"[radar] {key}: kept previous {blk.get('init')} loop")
        if blk:
            models[key] = blk
    if not models:
        return None
    return {"built": R.utcnow().isoformat(timespec="seconds"), "order": [k for k in MODEL_ORDER if k in models],
            "models": models, "extent": EXTENT}
