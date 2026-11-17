"""
Forecast "radar" loops for the situational awareness board: simulated composite reflectivity
on the usual NWS dBZ scale, with sea-level pressure isobars and H/L centres, over a regional map
wide enough to see systems coming.

  HRRR   3 km, hourly to 48 h   AWS noaa-hrrr-bdp-pds, then NOMADS hrrr/prod
  RRFS   3 km, hourly to 60-84 h  AWS noaa-rrfs-ops-pds, then NOMADS rrfs/prod

Only the needed GRIB records are downloaded (byte ranges from each file's .idx). Frames are
written to site/radar/ and described in meta["radar"]; a run whose model cycle is unchanged
reuses the previous frames from data/site_cache, and a model that fails keeps its last good loop.

NOTE: the .idx field strings follow the usual NCEP inventories (REFC:entire atmosphere,
MSLMA/PRMSL/MSLET:mean sea level). They could not be probed live from the build sandbox; the
Actions log prints the URLs tried and the fields each model actually matched.
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pygrib
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

import refs_qpf as R

HRRR_URLS = ["https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d:%Y%m%d}/conus/hrrr.t{d:%H}z.wrfsfcf{f:02d}.grib2",
             "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{d:%Y%m%d}/conus/hrrr.t{d:%H}z.wrfsfcf{f:02d}.grib2"]
RRFS_URLS = ["https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.prslev.3km.f{f:03d}.conus.grib2",
             "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.prslev.3km.f{f:03d}.conus.grib2",
             "https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.2dfld.3km.f{f:03d}.conus.grib2"]

MODELS = {
    # Only the 6-hourly synoptic cycles are tried (they are the long runs). lengths: longest first.
    "hrrr": {"label": "HRRR", "urls": HRRR_URLS, "lengths": [48, 36, 18], "latency_h": 2, "smooth": 8.0},
    "rrfs": {"label": "RRFS", "urls": RRFS_URLS, "lengths": [72, 60, 48, 36, 18], "latency_h": 2, "smooth": 8.0},
}
MODEL_ORDER = ["hrrr", "rrfs"]
WORKERS = 6
OUT_SUB = "radar"

_KEEP = re.compile(r":(REFC:entire atmosphere|MSLMA:mean sea level|PRMSL:mean sea level|MSLET:mean sea level)")

# NWS reflectivity scale
DBZ = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
DBZ_COLORS = ["#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00", "#fdf802", "#e5bc00",
              "#fd9500", "#fd0000", "#d40000", "#bc0000", "#f800fd", "#9854c6"]
CMAP = mcolors.ListedColormap(DBZ_COLORS)
NORM = mcolors.BoundaryNorm(DBZ, CMAP.N)

EXTENT = None                    # board.radar_extent from site_config.json
CITIES = []                      # board.radar_cities: [[name, lat, lon(, "left")], ...]
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
    print(f"[radar] {model}: nothing found; last URLs tried: {_urls(model, base - timedelta(hours=24), m['lengths'][-1])}")
    return None, None


def read_frame(model, init, f):
    path = R.download_subset(_urls(model, init, f), lambda l: bool(_KEEP.search(l)))
    out, crop = {}, None
    try:
        with pygrib.open(path) as g:
            for msg in g:
                key = {"refc": "refc", "mslma": "mslp", "prmsl": "mslp", "mslet": "mslp"}.get(msg.shortName.lower())
                if key is None or key in out:
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
def _halo(w=2.5):
    return [pe.withStroke(linewidth=w, foreground="#000000")]


class RadarRenderer:
    """One figure and basemap reused for every frame; only the data artists are redrawn."""

    def __init__(self, model):
        w, e, s, n = EXTENT
        aspect = (n - s) / ((e - w) * np.cos(np.radians((n + s) / 2)))
        W = 7.2
        H = W * aspect + 0.75
        self.fig = plt.figure(figsize=(W, H))
        top, bot = 0.36 / H, 0.39 / H
        self.ax = self.fig.add_axes([0.0, bot, 1.0, 1 - top - bot], projection=ccrs.PlateCarree())
        ax, pc = self.ax, ccrs.PlateCarree()
        ax.set_extent(EXTENT, crs=pc)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0e1b26", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#22313d", zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#0e1b26", edgecolor="#4d6476", linewidth=0.4, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#a9bccb", linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#6d8597", linewidth=0.5, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#6d8597", linewidth=0.5, zorder=4)
        for c in CITIES:
            name, la, lo = c[:3]
            left = len(c) > 3 and c[3] == "left"
            ax.plot(lo, la, marker="o", ms=3, mfc="#e6edf2", mec="#000000", mew=0.5, transform=pc, zorder=9)
            ax.text(lo + (-0.12 if left else 0.12), la + 0.08, name, fontsize=8, color="#ffffff", transform=pc,
                    zorder=9, ha="right" if left else "left", path_effects=_halo())
        plat, plon, _ = R.SITES[R.PRIMARY_SITE]
        ax.plot(plon, plat, marker="*", ms=13, mfc="#ffd84a", mec="#000000", mew=0.9, transform=pc, zorder=11)
        ax.text(plon + 0.2, plat, R.SITE_LABELS.get(R.PRIMARY_SITE, R.PRIMARY_SITE), fontsize=9, fontweight="bold",
                color="#ffd84a", va="center", transform=pc, zorder=11, path_effects=_halo(3))
        ax.spines["geo"].set_edgecolor("#4d6476")
        ty = 1 - 0.18 / H
        self.t1 = self.fig.text(0.012, ty, "", fontsize=12, fontweight="bold", color="#ffffff", va="center")
        self.t2 = self.fig.text(0.988, ty, "", fontsize=9.5, color="#c9d6e0", ha="right", va="center")
        cax = self.fig.add_axes([0.2, 0.13 / H, 0.6, 0.11 / H])
        cb = self.fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", ticks=DBZ)
        cb.ax.tick_params(labelsize=7.5, colors="#c9d6e0", length=2, pad=1)
        cb.outline.set_edgecolor("#4d6476")
        self.fig.text(0.81, 0.185 / H, "dBZ", fontsize=8.5, color="#c9d6e0", va="center")
        self.dyn = []
        self.smooth = MODELS[model]["smooth"]
        self.label = MODELS[model]["label"]

    def draw(self, fields, crop, init, f, path):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        ax, pc = self.ax, ccrs.PlateCarree()
        lats, lons = crop["lats"], crop["lons"]
        z = np.ma.masked_where(~np.isfinite(fields["refc"]) | (fields["refc"] < DBZ[0]), fields["refc"])
        if z.count():
            self.dyn.append(ax.pcolormesh(lons, lats, z, cmap=CMAP, norm=NORM, transform=pc, zorder=2,
                                          shading="auto", rasterized=True))
        if "mslp" in fields:
            self.dyn += self._isobars(fields["mslp"], lats, lons)
        valid = init + timedelta(hours=f)
        loc = valid.astimezone(TZ)
        self.t1.set_text(f"{loc:%a %b %-d, %-I %p %Z}  ({valid:%HZ})")
        self.t2.set_text(f"{self.label} simulated radar · hour {f} of the {init:%HZ %a} run")
        self.fig.savefig(path, dpi=150, facecolor="#0f1a24")
        R.quantize_png(path)

    def _isobars(self, p, lats, lons):
        ax, pc = self.ax, ccrs.PlateCarree()
        p = np.where(np.isfinite(p), p, np.nanmean(p))
        hpa = gaussian_filter(p / 100.0 if np.nanmax(p) > 2000 else p, self.smooth)
        cs = ax.contour(lons, lats, hpa, levels=np.arange(960, 1052, 4), colors="#ffffff", linewidths=0.9,
                        alpha=0.8, transform=pc, zorder=5)
        ax.clabel(cs, fmt="%d", fontsize=7.5, inline=True, inline_spacing=2, colors="#ffffff")
        out = [cs]                                   # ContourSet.remove() also removes its labels
        size = max(5, int(round(min(hpa.shape) / 5)))
        for kind, filt, color in (("H", maximum_filter, "#4aa3ff"), ("L", minimum_filter, "#ff5a5a")):
            ext = (hpa == filt(hpa, size=size, mode="nearest"))
            edge = max(2, size // 3)
            ext[:edge, :] = ext[-edge:, :] = False
            ext[:, :edge] = ext[:, -edge:] = False
            for j, i in zip(*np.where(ext)):
                v = hpa[j, i]
                if (kind == "H" and v < 1012) or (kind == "L" and v > 1012):
                    continue
                out.append(ax.text(lons[j, i], lats[j, i], kind, fontsize=16, fontweight="bold", color=color,
                                   ha="center", va="center", transform=pc, zorder=7, path_effects=_halo(3)))
        return out


# --------------------------------------------------------------------------- build
def build_model(model, out_dir, previous=None):
    m = MODELS[model]
    init, length = find_init(model)
    if not init:
        raise FileNotFoundError(f"{model}: no recent cycle with a posted f{m['lengths'][-1]:03d}")
    hours = list(range(1, length + 1))
    sub = os.path.join(out_dir, OUT_SUB)
    os.makedirs(sub, exist_ok=True)
    if previous and previous.get("init") == init.isoformat() and len(previous.get("frames", [])) >= len(hours) * 0.9 \
            and all(os.path.exists(os.path.join(R.SITE_CACHE, fr["img"])) for fr in previous["frames"]):
        for fr in previous["frames"]:
            shutil.copy(os.path.join(R.SITE_CACHE, fr["img"]), os.path.join(out_dir, fr["img"]))
        print(f"[radar] {model} {init:%d/%HZ} unchanged; reused {len(previous['frames'])} frames")
        return previous
    print(f"[radar] {model} {init:%Y%m%d %HZ}: {len(hours)} frames f01-f{length:02d}  ({_urls(model, init, 1)[0]})")

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
    return {"label": m["label"], "note": "hourly", "init": init.isoformat(), "frames": frames}


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
