"""
Event look-back: how REFS, HRRR and RRFS forecast a past rainfall window,
verified against MRMS MultiSensor Pass2 QPE.

  python lookback.py --end 2026110312 --hours 24 --title "Melbourne / Palm Bay flooding"

Data
  MRMS  {hours}H Pass2 ending at --end (NCEP while fresh, then AWS noaa-mrms-pds)
  REFS  LPMM hourly from every 6-hourly run whose f60 covers the window
        (NOMADS while fresh, then AWS noaa-rrfs-ops-pds, which starts 2026-08-13);
        AVRG from the shortest-lead run
  HRRR  run-total APCP difference, 00/06/12/18Z runs to f48 (AWS noaa-hrrr-bdp-pds)
  RRFS  deterministic 2dfld APCP, 00/06/12/18Z runs to f84 (AWS noaa-rrfs-ops-pds)

Output: data/cases/<id>/{case.json, refs.png, models.png, dprog.png}
"""
import argparse
import glob
import os
import shutil
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygrib

import refs_lpmm_cape as R

HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d:%Y%m%d}/conus/hrrr.t{d:%H}z.wrfsfcf{f:02d}.grib2"
RRFS_URL = "https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.2dfld.3km.f{f:03d}.conus.grib2"
REFS_AWS_START = datetime(2026, 8, 13, tzinfo=timezone.utc)

ALLOWED_HOURS = (6, 12, 24, 48, 72)
MAX_LEAD = 36                   # h before window start; keeps the dprog/dt chart readable
EVENT_RADIUS_KM = 20.0          # "how close did the max get" search radius
CASE_THRESHOLDS = [1.0, 2.0, 4.0]
MODEL_MAX_FHR = {"REFS": 60, "HRRR": 48, "RRFS": 84}
MODEL_COLORS = {"REFS": R.RUN_COLORS[0], "AVRG": R.AVRG_COLOR, "HRRR": "#f0b429", "RRFS": "#5fd38d"}


def lead_inits(model, start, end):
    """Inits (newest first) with lead 0..MAX_LEAD h whose forecast reaches the window end."""
    out = []
    for lead in range(0, MAX_LEAD + 1, 6):
        init = start - timedelta(hours=lead)
        if (end - init).total_seconds() / 3600 <= MODEL_MAX_FHR[model]:
            out.append((lead, init))
    return out

# ---------------------------------------------------------------- model readers
def refs_window(init, ptype, start, end):
    f0 = int((start - init).total_seconds() // 3600)
    f1 = int((end - init).total_seconds() // 3600)
    if init < REFS_AWS_START and R.utcnow() - init > timedelta(hours=40):
        print(f"  REFS {init:%d/%HZ}: before AWS archive start, skipped")
        return None
    data = R.fetch_qpf_hours(init, ptype, list(range(f0 + 1, f1 + 1)))
    return None if data is None else (data.sum(axis=0), R._GRID["lats"], R._GRID["lons"])


def _apcp_total(url, fhr):
    """0-fhr run-total APCP (mm, cropped) from a deterministic file via byte-range."""
    if fhr == 0:
        return None
    path = R.download_subset([url], lambda l: ":APCP:surface:" in l)
    try:
        with pygrib.open(path) as g:
            msgs = [m for m in g if R._is_apcp(m)]
            R._log_inventory(f"apcp-{url.split('/')[2].split('.')[0]} f{fhr}", msgs)
            pick = next((m for m in msgs if R._step_range(m) == (0, fhr)), None)
            if pick is None:
                raise ValueError(f"no 0-{fhr} h APCP: {[R._step_range(m) for m in msgs]}")
            crop = R.make_crop(pick)
            return R.crop_values(pick, crop), crop
    finally:
        os.remove(path)


def det_window(model, init, start, end):
    tmpl = HRRR_URL if model == "HRRR" else RRFS_URL
    f0 = int((start - init).total_seconds() // 3600)
    f1 = int((end - init).total_seconds() // 3600)
    try:
        b, crop = _apcp_total(tmpl.format(d=init, f=f1), f1)
        a = _apcp_total(tmpl.format(d=init, f=f0), f0) if f0 > 0 else None
    except Exception as e:
        print(f"  ! {model} {init:%d/%HZ}: {e}")
        return None
    tot = (b - (a[0] if a else 0)) * R.MM_TO_IN
    return np.clip(tot, 0, None), crop["lats"], crop["lons"]

# ---------------------------------------------------------------- scoring
def evaluate(field, lats, lons, raw, event_pt):
    obs = R.sample_mrms(raw, lats, lons)
    mask = R.in_view(lats, lons) & np.isfinite(obs)
    f, o = field[mask], obs[mask]
    near = R.dist_km(lats, lons, *event_pt) <= EVENT_RADIUS_KM
    row = {"domain_max": R.r2(np.where(R.in_view(lats, lons), field, -1).max()),
           "near_event_max": R.r2(field[near].max()) if near.any() else None,
           "bias_ratio": R.r2(f.sum() / o.sum()) if o.sum() > 0.01 * o.size else None}
    for t in CASE_THRESHOLDS:
        row[f"ets_{t:g}"], row[f"fbias_{t:g}"] = R.categorical(f, o, t)
    sites = {}
    for name, (la, lo, _) in R.SITES.items():
        d = R.dist_km(lats, lons, la, lo)
        j, i = np.unravel_index(np.argmin(d), d.shape)
        sites[name] = R.r2(field[j, i])
    row["sites"] = sites
    return row

# ---------------------------------------------------------------- plotting
def _panel(ax, field, lats, lons, title, color, event_pt, labels=False):
    R.add_basemap(ax, labels=labels)
    cs = R.shade(ax, field, lats=lats, lons=lons)
    R.annotate_max(ax, field, lats=lats, lons=lons)
    ax.plot(event_pt[1], event_pt[0], "o", mfc="none", mec="#ffffff", ms=11, mew=1.4,
            transform=ccrs.PlateCarree(), zorder=8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=color)
    return cs


def plot_grid(panels, out, suptitle, event_pt):
    n = len(panels)
    cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.0 * rows + 0.9),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.11 if rows > 1 else 0.18, top=0.9, wspace=0.04, hspace=0.14)
    cs = None
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        fld, la, lo, title, col = panels[k]
        cs = _panel(ax, fld, la, lo, title, col, event_pt, labels=(k == 0))
    R.colorbar(fig, cs, [0.18, 0.04 if rows > 1 else 0.08, 0.64, 0.018 if rows > 1 else 0.03],
               "Precipitation (in)  ·  ○ = observed max", R.CLEVS)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.975)
    fig.savefig(out, dpi=125)
    plt.close(fig)


def plot_dprog(rows, obs_max, obs_max_3km, out, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.22)
    for model in ("REFS", "HRRR", "RRFS"):
        rr = sorted([r for r in rows if r["model"] == model], key=lambda r: r["lead"])
        if not rr:
            continue
        x = [r["lead"] for r in rr]
        a1.plot(x, [r["near_event_max"] for r in rr], "-o", ms=4, color=MODEL_COLORS[model], label=model)
        a2.plot(x, [r["ets_1"] for r in rr], "-o", ms=4, color=MODEL_COLORS[model], label=f"{model} ≥ 1\"")
        a2.plot(x, [r["ets_2"] for r in rr], ":", marker="o", ms=3, color=MODEL_COLORS[model], label=f"{model} ≥ 2\"")
    a1.axhline(obs_max, color=R.TEXT, ls="--", lw=1, label=f"MRMS max {obs_max:.2f}\"")
    a1.axhline(obs_max_3km, color=R.MUTED, ls=":", lw=1, label=f"MRMS 3-km max {obs_max_3km:.2f}\"")
    a1.set_ylabel(f"Max within {EVENT_RADIUS_KM:g} km of observed max (in)")
    a1.set_ylim(bottom=0)
    a2.axhline(0, color=R.MUTED, lw=0.6)
    a2.set_ylabel("Domain ETS")
    for a in (a1, a2):
        a.set_xlabel("Lead before window start (h)")
        a.invert_xaxis()
        a.grid(color="#263848", lw=0.6)
        a.legend(frameon=False, fontsize=8.5, loc="upper left")
        for sp in a.spines.values():
            sp.set_color(R.LINE)
    a1.set_title("Did they get the peak?", fontsize=11, fontweight="bold")
    a2.set_title("Did they get the placement?", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    fig.savefig(out, dpi=130)
    plt.close(fig)

# ---------------------------------------------------------------- build
def nearest_site(lat, lon):
    best = min(R.SITES.items(), key=lambda kv: R.dist_km(np.array(kv[1][0]), np.array(kv[1][1]), lat, lon))
    d = float(R.dist_km(np.array(best[1][0]), np.array(best[1][1]), lat, lon))
    return best[0].split(" (")[0], round(d)


def build_case(end, hours=24, title=None, auto=False, case_id=None):
    if hours not in ALLOWED_HOURS:
        raise ValueError(f"hours must be one of {ALLOWED_HOURS}")
    end = end.replace(minute=0, second=0, microsecond=0)
    if end.hour % 6:
        raise ValueError("end must be 00, 06, 12 or 18Z")
    start = end - timedelta(hours=hours)
    case_id = case_id or (f"auto-{end:%Y%m%d%H}" if auto else f"{end:%Y%m%d%H}-{hours}h")
    out_dir = os.path.join(R.CASES_DIR, case_id)
    if auto and os.path.exists(os.path.join(out_dir, "case.json")):
        print(f"[case] {case_id} already built")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    title = title or f"{hours} h ending {end:%HZ %a %d %b %Y}"
    print(f"[case] {case_id}: {start:%Y%m%d %HZ} to {end:%Y%m%d %HZ}")
    notes = []

    raw = R.fetch_mrms_raw(hours, end)
    if raw is None:
        raise RuntimeError("MRMS QPE unavailable for this window")
    obs, olat, olon = R.mrms_native(raw)
    view = R.in_view(olat, olon) & np.isfinite(obs)
    jm, im = np.unravel_index(np.argmax(np.where(view, obs, -1)), obs.shape)
    event_pt = (float(olat[jm, im]), float(olon[jm, im]))
    obs_max = float(obs[jm, im])

    rows, refs_panels, best = [], [], {}
    for lead, init in lead_inits("REFS", start, end):
        got = refs_window(init, "lpmm", start, end)
        if got is None:
            notes.append(f"REFS {init:%d/%HZ} unavailable")
            continue
        fld, la, lo = got
        rows.append(dict(evaluate(fld, la, lo, raw, event_pt), model="REFS", init=init.isoformat(), lead=lead))
        refs_panels.append((fld, la, lo, f"REFS LPMM  ·  {lead} h lead ({init:%d/%HZ})", MODEL_COLORS["REFS"]))
        best.setdefault("REFS", (fld, la, lo, lead, init))
    if "REFS" in best:
        _, _, _, lead, init = best["REFS"]
        got = refs_window(init, "avrg", start, end)
        if got:
            fld, la, lo = got
            rows.append(dict(evaluate(fld, la, lo, raw, event_pt), model="AVRG", init=init.isoformat(), lead=lead))
            best["AVRG"] = (fld, la, lo, lead, init)
    for model in ("HRRR", "RRFS"):
        for lead, init in lead_inits(model, start, end):
            got = det_window(model, init, start, end)
            if got is None:
                notes.append(f"{model} {init:%d/%HZ} unavailable")
                continue
            fld, la, lo = got
            rows.append(dict(evaluate(fld, la, lo, raw, event_pt), model=model, init=init.isoformat(), lead=lead))
            best.setdefault(model, (fld, la, lo, lead, init))

    o3 = R.sample_mrms(raw, olat[::3, ::3], olon[::3, ::3])
    obs_max_3km = float(np.nanmax(np.where(R.in_view(olat[::3, ::3], olon[::3, ::3]), o3, np.nan)))
    win_txt = f"{start:%HZ %a %d %b} to {end:%HZ %a %d %b %Y}"
    obs_panel = (obs, olat, olon, f"MRMS {hours}-h QPE (Pass 2)", R.TEXT)

    plot_grid([obs_panel] + refs_panels, os.path.join(out_dir, "refs.png"),
              f"{title}\nREFS LPMM by lead time vs MRMS, {win_txt}", event_pt)
    mp = [obs_panel] + [(b[0], b[1], b[2], f"{m}  ·  {b[3]} h lead ({b[4]:%d/%HZ})", MODEL_COLORS[m])
                        for m, b in best.items()]
    plot_grid(mp, os.path.join(out_dir, "models.png"),
              f"{title}\nShortest-lead forecast from each model vs MRMS, {win_txt}", event_pt)
    if rows:
        plot_dprog(rows, obs_max, obs_max_3km, os.path.join(out_dir, "dprog.png"),
                   f"{title}: forecast trend as the event approached")

    obs_sites = {}
    for name, (la, lo, _) in R.SITES.items():
        v = R.sample_mrms(raw, np.array([[la]]), np.array([[lo]]))[0, 0]
        obs_sites[name] = R.r2(v)
    site_name, site_km = nearest_site(*event_pt)
    refs_best = min([r for r in rows if r["model"] == "REFS"], key=lambda r: r["lead"], default=None)
    headline = None
    if refs_best and refs_best["near_event_max"] is not None:
        headline = (f"MRMS max {obs_max:.2f}″ near {site_name}; shortest-lead REFS LPMM reached "
                    f"{refs_best['near_event_max']:.2f}″ within {EVENT_RADIUS_KM:g} km")
    case = {
        "id": case_id, "title": title, "auto": auto, "hours": hours,
        "start": start.isoformat(), "end": end.isoformat(), "built": R.utcnow().isoformat(timespec="seconds"),
        "obs": {"max": R.r2(obs_max), "max_3km": R.r2(obs_max_3km), "lat": round(event_pt[0], 3),
                "lon": round(event_pt[1], 3), "near": site_name, "near_km": site_km, "sites": obs_sites},
        "headline": headline, "thresholds": CASE_THRESHOLDS, "event_radius_km": EVENT_RADIUS_KM,
        "rows": rows, "notes": notes,
        "images": [f for f in ("refs.png", "models.png", "dprog.png") if os.path.exists(os.path.join(out_dir, f))],
    }
    R.save_json(os.path.join(out_dir, "case.json"), case, indent=1)
    print(f"[case] {case_id} built: {len(rows)} forecasts, MRMS max {obs_max:.2f}\" near {site_name}")
    return out_dir


def prune_auto(keep):
    autos = sorted(glob.glob(os.path.join(R.CASES_DIR, "auto-*")))
    for d in autos[:-keep] if keep else autos:
        shutil.rmtree(d, ignore_errors=True)


def republish():
    """Rebuild ./site from the cached last site plus all cases (look-back workflow)."""
    if not os.path.isdir(R.SITE_CACHE):
        print("[case] no cached site yet; run the main workflow once first")
        return False
    if os.path.isdir(R.OUT_DIR):
        shutil.rmtree(R.OUT_DIR)
    shutil.copytree(R.SITE_CACHE, R.OUT_DIR)
    if os.path.exists(R.INDEX_SRC):
        shutil.copy(R.INDEX_SRC, os.path.join(R.OUT_DIR, "index.html"))
    meta_p = os.path.join(R.OUT_DIR, "meta.json")
    meta = R.load_json(meta_p, {})
    n = R.publish_cases(R.OUT_DIR)
    meta.setdefault("products", {})["cases"] = n > 0
    R.save_json(meta_p, meta, indent=1)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", required=True, help="window end, YYYYMMDDHH UTC (00/06/12/18)")
    ap.add_argument("--hours", type=int, default=24, choices=ALLOWED_HOURS)
    ap.add_argument("--title", default="")
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args()
    end = datetime.strptime(a.end, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    build_case(end, a.hours, a.title or None)
    if not a.no_publish:
        republish()
