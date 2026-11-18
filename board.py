"""
Situational awareness board: WPC Probabilistic Precipitation Portal (low / expected / high),
WPC Excessive Rainfall Outlook (Days 1-5), the local NWS Area Forecast Discussion, and a
rule-based action list that cites the source of every item.

Sources (checked 2026-11-17)
  PPP   https://ftp-wpc.ncep.noaa.gov/prob_precip_portal/co/  (latest_cycle.txt; ppp_co_{24,48,72}hr_{cycle}/
        ndfd_*p{acc}i_* = expected (official NWS), *pp{acc}ip0p10_* = low end, *pp{acc}ip0p90_* = high end)
  ERO   https://mapservices.weather.noaa.gov/vector/rest/services/hazards/wpc_precip_hazards/MapServer/{0..4}
        fields: dn (1 MRGL, 2 SLGT, 3 MDT, 4 HIGH), outlook, issue_time, start_time, end_time, valid_time
  AFD   https://api.weather.gov/products/types/AFD/locations/{WFO}, then tgftp raw text as fallback

The optional AI summary runs only when an ANTHROPIC_API_KEY secret is present; the rule-based
actions never depend on it.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygrib

import refs_qpf as R

PPP_ROOTS = ["https://ftp-wpc.ncep.noaa.gov/prob_precip_portal/co", "https://ftp.wpc.ncep.noaa.gov/prob_precip_portal/co"]
PPP_ACCUMS = [24, 48, 72]
PPP_PRODUCTS = {"low": "ip0p10", "high": "ip0p90"}      # expected is the NDFD "p{acc}i" file
ERO_SERVICE = "https://mapservices.weather.noaa.gov/vector/rest/services/hazards/wpc_precip_hazards/MapServer"
ERO_DAYS = [1, 2, 3, 4, 5]
ERO_NAMES = {0: "None", 1: "Marginal", 2: "Slight", 3: "Moderate", 4: "High"}
ERO_PCT = {1: "≥5%", 2: "≥15%", 3: "≥40%", 4: "≥70%"}
ERO_COLORS = {1: "#38a800", 2: "#fffe00", 3: "#f50000", 4: "#ff69c5"}
AFD_API = "https://api.weather.gov/products/types/AFD/locations/{wfo}"
AFD_TGFTP = ["https://tgftp.nws.noaa.gov/data/raw/fx/fxus{reg}.k{w}.afd.{w}.txt",
             "https://tgftp.ncep.noaa.gov/data/raw/fx/fxus{reg}.k{w}.afd.{w}.txt"]
ANTHROPIC_MODEL = "claude-sonnet-5-5"

# AFD keyword rules: (regex, severity, action). Severity 3 = act now, 2 = prepare, 1 = monitor.
AFD_RULES = [
    (r"flash flood (warning|emergency)", 3, "Flash flood warning referenced: confirm warning polygons and activate flash flood response."),
    (r"flood watch", 3, "Flood Watch referenced: review flood response checklists and pre-stage pumps and barricades."),
    (r"major flood stage|record flood", 3, "River(s) at or forecast to major flood stage: coordinate road closures and check river forecasts."),
    (r"river flood|flood stage|flood warning", 2, "River flooding ongoing or forecast: monitor river forecasts and affected crossings."),
    (r"excessive rainfall|heavy rain(fall)?|flash flood", 2, "Heavy rain potential noted: check low-water crossings, drainage and pump readiness."),
    (r"tornado", 3, "Tornado potential mentioned: verify shelter plans and warning dissemination."),
    (r"severe|damaging wind|hail", 2, "Strong to severe storms possible: secure loose equipment and review lightning/wind safety."),
    (r"rip current|rough surf|high surf|beach erosion|hazardous beach", 1, "Beach and surf hazards: notify beach operations and lifeguard coordinators."),
    (r"small craft|hazardous boating|poor to hazardous boating|gale", 1, "Marine hazards: brief boating and waterside operations."),
    (r"wind chill|freeze|frost", 1, "Cold temperatures: check cold-weather shelter and sensitive-equipment protection."),
]


def board_sites():
    """Sites shown on the board: every site except those listed in board.exclude_sites."""
    return {n: v for n, v in R.SITES.items() if n not in R.BOARD_EXCLUDE}


def _get(url, **kw):
    return R.SESSION.get(url, timeout=kw.pop("timeout", 30), **kw)


# --------------------------------------------------------------------------- PPP
def ppp_latest_cycle():
    for root in PPP_ROOTS:
        try:
            r = _get(f"{root}/latest_cycle.txt")
            if r.status_code == 200 and r.text.strip():
                cyc = r.text.split()[0]
                return datetime.strptime(cyc, "%Y%m%d%H").replace(tzinfo=timezone.utc), root
            print(f"[ppp] {root}/latest_cycle.txt: HTTP {r.status_code}")
        except Exception as e:
            print(f"[ppp] {root}: {e}")
    return None, None


def _ppp_listing(root, acc, cyc):
    """File names in one PPP GRIB2 directory, read from the server's own index page."""
    url = f"{root}/ppp_co_{acc}hr_{cyc:%Y%m%d%H}/"
    r = _get(url)
    r.raise_for_status()
    return url, sorted(set(re.findall(r'href="([^"/?]+\.grib2)"', r.text)))


def _ppp_pattern(acc, tag):
    """tag: 'expected' (NDFD p{acc}i file) or an 'ip0pNN' percentile code."""
    return re.compile(rf"^ndfd_(?:co_)?p{acc}i_(\d{{10}})f(\d{{3}})\.grib2$") if tag == "expected" else \
        re.compile(rf"^ndfd_(?:co_)?pp{acc}{tag}_(\d{{10}})f(\d{{3}})\.grib2$")


def _hours(files, acc, tag):
    """{forecast hour: file name} for one product in a directory listing."""
    pat = _ppp_pattern(acc, tag)
    return {int(m.group(2)): f for f in files for m in [pat.match(f)] if m}


def _pick(files, acc, tag):
    """File whose window starts closest to the cycle (so the 24-h tab is the next 24 h, not a mostly
    past window that happens to end soonest)."""
    hits = _hours(files, acc, tag)
    return hits[min(hits, key=lambda f: (abs(f - acc), f))] if hits else None


def _day_hours(files, cyc):
    """Forecast hours ending Days 1-3: three back-to-back 24-h periods starting at the forecast time.
    The portal's 24-h windows end 23-71 h after the cycle (f023, f029 ... f071), so only windows
    measured from the cycle give a full Day 3 every run; 12Z-12Z days fit just 2 days from most
    cycles. Day 1 ends at the first 24-h window ending >= 23 h out, then +24 h and +48 h."""
    have = set(_hours(files, 24, "expected"))
    first = min((f for f in have if f >= 23), default=None)
    if first is None:
        return []
    return [f for f in (first, first + 24, first + 48) if f in have]


def _decode_ppp(path):
    with pygrib.open(path) as g:
        m = g.message(1)
        units = str(getattr(m, "units", ""))
        crop = R.make_crop(m)
        vals = np.ma.filled(m.values.astype(np.float64), np.nan)[crop["rows"], crop["cols"]]
    vals = np.where((vals >= 0) & (vals < 5000), vals, np.nan)
    return vals.astype(np.float32), crop, units


def _to_inches(fields):
    """Pick one unit for all fields of an accumulation. GRIB units say so when they can; otherwise the
    wettest field decides (a 99th percentile above 25 can only be mm). Deciding per field would
    convert a dry low-end field differently from a wet high-end one."""
    units = " ".join(u for _, _, u in fields.values()).lower()
    if "kg" in units or "mm" in units:
        mm = True
    elif re.search(r"\bin(ch|ches)?\b", units):
        mm = False
    else:
        wet = max((np.nanpercentile(v[v > 0], 99) for v, _, _ in fields.values() if np.any(v > 0)), default=0)
        mm = wet > 25
    return {k: ((v * R.MM_TO_IN if mm else v).astype(np.float32), c, u) for k, (v, c, u) in fields.items()}, ("mm" if mm else "in")


def _ppp_window(url, files, acc, picks, cyc, label):
    """Download and decode one window's expected/low/high files -> accums entry, or None."""
    print(f"[ppp] {label}: " + ", ".join(f"{k}={v}" for k, v in picks.items()))
    fields, fh = {}, None
    for k, fname in picks.items():
        if not fname:
            continue
        try:
            p = R.download_any([url + fname])
            try:
                fields[k] = _decode_ppp(p)
            finally:
                os.remove(p)
            fh = int(re.search(r"f(\d{3})\.grib2$", fname).group(1))
        except Exception as e:
            print(f"[ppp] {fname}: {e}")
    if not fields:
        return None
    fields, unit = _to_inches(fields)
    end = cyc + timedelta(hours=fh)
    for k, (v, _, u) in fields.items():
        print(f"[ppp]   {label} {k}: units {u!r} -> {unit}, domain max {np.nanmax(v):.2f} in")
    start = end - timedelta(hours=acc)
    w = {"fields": fields, "hours": acc, "start": start.isoformat(), "end": end.isoformat(),
         "past_h": max(0, int((cyc - start).total_seconds() // 3600))}
    w["exceed"] = R.optional(f"ppp exceedance {label}", _exceedance, url, files, acc, fh, label) or {}
    return w


_EXC = r"^ndfd_(?:co_)?pp{acc}ige(\d+)p(\d+)_\d{{10}}f{fh:03d}\.grib2$"


def _exceedance(url, files, acc, fh, label):
    """Chance (%) of at least each amount the portal publishes (0.01" ... 18"), sampled at the board
    sites only: {site: [[inches, pct], ...]} sorted by amount. Amounts come from the file names,
    so nothing is assumed about which thresholds exist."""
    from concurrent.futures import ThreadPoolExecutor
    pat = re.compile(_EXC.format(acc=acc, fh=fh))
    todo = sorted((float(f"{m.group(1)}.{m.group(2)}"), f) for f in files for m in [pat.match(f)] if m)
    if not todo:
        print(f"[ppp]   {label}: no exceedance files at f{fh:03d}")
        return None
    sites = board_sites()

    def one(item):
        thr, fname = item
        try:
            p = R.download_any([url + fname])
            try:
                vals, crop, _ = _decode_ppp(p)
            finally:
                os.remove(p)
        except Exception as e:
            print(f"[ppp]   {fname}: {e}")
            return thr, None
        out = {}
        for n, (la, lo, _) in sites.items():
            d = R.dist_km(crop["lats"], crop["lons"], la, lo)
            j, i = np.unravel_index(np.argmin(d), d.shape)
            out[n] = float(vals[j, i]) if np.isfinite(vals[j, i]) else None
        return thr, out

    with ThreadPoolExecutor(6) as ex:
        got = [g for g in ex.map(one, todo) if g[1] is not None]
    if not got:
        return None
    top = max((v for _, o in got for v in o.values() if v is not None), default=0)
    scale = 100.0 if top <= 1.0001 else 1.0                       # fraction or percent, decided once
    res = {n: [[thr, round(min(100.0, max(0.0, o[n] * scale)))] for thr, o in got if o.get(n) is not None]
           for n in sites}
    print(f"[ppp]   {label}: exceedance at {len(got)} amounts ({got[0][0]:g}-{got[-1][0]:g} in)")
    return res


def fetch_ppp():
    """Windows 'd1'..'d3' (consecutive 24-h days from the forecast time) and '48' / '72', which start when Day 1 starts and
    end when Day 2 / Day 3 end, so Day 1 + Day 2 covers exactly the 48-h window. (Only the
    expected amounts add up that way; 10th/90th percentiles of a sum are not sums of percentiles.)"""
    cyc, root = ppp_latest_cycle()
    if not cyc:
        return None
    out = {"cycle": cyc.isoformat(), "accums": {}}
    listings = {}
    for acc in PPP_ACCUMS:
        try:
            listings[acc] = _ppp_listing(root, acc, cyc)
            print(f"[ppp] {acc} h directory: {len(listings[acc][1])} files, expected-amount hours "
                  f"{sorted(_hours(listings[acc][1], acc, 'expected'))}")
        except Exception as e:
            print(f"[ppp] {acc} h listing failed: {e}")

    def window(acc, fh, label):
        url, files = listings[acc]
        picks = {"expected": _hours(files, acc, "expected").get(fh)}
        picks.update({k: _hours(files, acc, code).get(fh) for k, code in PPP_PRODUCTS.items()})
        return _ppp_window(url, files, acc, picks, cyc, label)

    days = _day_hours(listings[24][1], cyc) if 24 in listings else []
    print(f"[ppp] day-ending hours {days}")
    for d, fh in enumerate(days, 1):
        w = window(24, fh, f"Day {d} (f{fh:03d})")
        if w:
            w["day"] = d
            out["accums"][f"d{d}"] = w
    for acc, nd in ((48, 2), (72, 3)):
        if acc not in listings:
            continue
        have = _hours(listings[acc][1], acc, "expected")
        target = days[nd - 1] if len(days) >= nd else None
        if target in have:
            w = window(acc, target, f"{acc} h = Days 1-{nd} (f{target:03d})")
            if w:
                w["days"] = nd
        else:                                   # can't line up with the days: nearest window, flagged
            fh = min(have, key=lambda f: (abs(f - acc), f)) if have else None
            print(f"[ppp] {acc} h: no window ending with Day {nd} (f{target}); using f{fh}")
            w = window(acc, fh, f"{acc} h (f{fh:03d})") if fh is not None else None
        if w:
            out["accums"][str(acc)] = w
    _check_sums(out["accums"])
    return out if out["accums"] else None


def _check_sums(acc):
    """Log how well Day 1 + Day 2 (+ Day 3) match the 48 / 72-h expected maps over the domain."""
    for tot, parts in (("48", ["d1", "d2"]), ("72", ["d1", "d2", "d3"])):
        if tot not in acc or not all(p in acc for p in parts) or not acc[tot].get("days"):
            continue
        try:
            t = acc[tot]["fields"]["expected"][0]
            s = sum(acc[p]["fields"]["expected"][0] for p in parts)
            if t.shape != s.shape:
                continue
            d = np.abs(np.nan_to_num(t) - np.nan_to_num(s))
            print(f"[ppp] check {tot} h vs {'+'.join(parts)} expected: max diff {np.nanmax(d):.2f} in, "
                  f"mean {np.nanmean(d):.3f} in")
        except Exception as e:
            print(f"[ppp] check {tot}: {e}")


def ppp_site_values(ppp):
    sites = {}
    for name, (la, lo, _) in board_sites().items():
        per = {}
        for acc, a in ppp["accums"].items():
            v = {}
            for k, (vals, crop, _) in a["fields"].items():
                d = R.dist_km(crop["lats"], crop["lons"], la, lo)
                j, i = np.unravel_index(np.argmin(d), d.shape)
                v[k] = R.r2(vals[j, i]) if np.isfinite(vals[j, i]) else None
            if a.get("exceed", {}).get(name):
                v["exceed"] = a["exceed"][name]
            per[acc] = v
        sites[name] = per
    return sites


PPP_ORDER = ["d1", "d2", "d3", "48", "72"]
OUTLINE_LEVS = [7, 10, 15, 20]      # where WPC's orange / gold / yellow steps look alike on screen


def _outline(ax, vals, crop):
    """Thin labeled lines at the 7, 10, 15 and 20-inch steps so neighbouring colours are easy to tell apart."""
    v = np.nan_to_num(vals)
    levs = [l for l in OUTLINE_LEVS if np.nanmax(v) >= l]
    if not levs:
        return
    cs = ax.contour(crop["lons"], crop["lats"], v, levels=levs, colors="#111111", linewidths=0.8,
                    transform=ccrs.PlateCarree(), zorder=3)
    ax.clabel(cs, fmt=lambda x: f'{x:g}"', fontsize=8, inline=True, inline_spacing=1, colors="#111111")


def ppp_label(key, a):
    if a.get("day"):
        return f"Day {a['day']} (24-h)"
    return f"{a['hours']}-h total (Days 1-{a['days']})" if a.get("days") else f"{a['hours']}-h total"


def plot_ppp(ppp, key, out_dir):
    a = ppp["accums"].get(key)
    if not a:
        return None
    order = [k for k in ("low", "expected", "high") if k in a["fields"]]
    titles = {"low": "Low end (10th percentile)", "expected": "Expected (official NWS)", "high": "High end (90th percentile)"}
    fig, axes = plt.subplots(1, len(order), figsize=(5.4 * len(order), 6.6), subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, (fk, ax) in enumerate(zip(order, axes[0])):
        vals, crop, _ = a["fields"][fk]
        R.add_basemap(ax, labels=(k == 0))
        cs = R.shade(ax, vals, kind="wpc", lats=crop["lats"], lons=crop["lons"])
        _outline(ax, vals, crop)
        R.annotate_max(ax, np.nan_to_num(vals), lats=crop["lats"], lons=crop["lons"])
        ax.set_title(titles[fk], fontsize=12, fontweight="bold")
    R.colorbar(fig, cs, [0.1, 0.1, 0.8, 0.03], "Precipitation (in), WPC color scale", R.WPC_CLEVS)
    s, e = datetime.fromisoformat(a["start"]), datetime.fromisoformat(a["end"])
    fig.suptitle(f"NWS Probabilistic Precipitation Portal, {ppp_label(key, a)} QPF\n{s:%HZ %a %d %b} to {e:%HZ %a %d %b}"
                 f"  ·  cycle {datetime.fromisoformat(ppp['cycle']):%d/%HZ}", fontsize=14, fontweight="bold", y=0.97)
    fname = f"ppp_{key}.png" if key.startswith("d") else f"ppp_{key}h.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    # where each map panel sits in the image (fractions, top-left origin) so the board can put
    # clickable site markers over it
    panels = []
    for ax in axes[0]:
        b = ax.get_position()
        x0, x1, y0, y1 = ax.get_extent(crs=ccrs.PlateCarree())
        panels.append({"box": [round(b.x0, 5), round(1 - b.y1, 5), round(b.x1, 5), round(1 - b.y0, 5)],
                       "extent": [round(x0, 4), round(x1, 4), round(y0, 4), round(y1, 4)]})
    plt.close(fig)
    R.quantize_png(os.path.join(out_dir, fname))
    return fname, panels


# --------------------------------------------------------------------------- ERO
def fetch_ero():
    out = {}
    for day in ERO_DAYS:
        try:
            r = _get(f"{ERO_SERVICE}/{day - 1}/query",
                     params={"where": "1=1", "outFields": "*", "outSR": "4326", "f": "geojson"})
            r.raise_for_status()
            out[day] = r.json().get("features", [])
        except Exception as e:
            print(f"[ero] day {day}: {e}")
    print(f"[ero] features per day: " + ", ".join(f"D{d}={len(f)}" for d, f in out.items()))
    return out


def _ero_meta(feats):
    for f in feats:
        p = f.get("properties", {})
        return {k: p.get(k) for k in ("issue_time", "start_time", "end_time", "valid_time")}
    return {}


def ero_summary(ero):
    from shapely.geometry import Point, box, shape
    view = box(R.EXTENT[0], R.EXTENT[2], R.EXTENT[1], R.EXTENT[3])
    days, sites = {}, {n: {} for n in board_sites()}
    for day, feats in ero.items():
        polys = []
        for f in feats:
            try:
                polys.append((int(f["properties"].get("dn") or 0), shape(f["geometry"])))
            except Exception:
                continue
        top = max([dn for dn, g in polys if g.intersects(view)], default=0)
        days[str(day)] = {"max_in_view": top, "name": ERO_NAMES.get(top, str(top)), **_ero_meta(feats)}
        for name, (la, lo, _) in board_sites().items():
            pt = Point(lo, la)
            sites[name][str(day)] = max([dn for dn, g in polys if g.contains(pt)], default=0)
    return {"days": days, "sites": sites}


def plot_ero(ero, out_dir):
    """One map per ERO day (1-5) so the board can lay them out side by side at full size."""
    from shapely.geometry import shape
    out = {}
    for day in ERO_DAYS:
        if day not in ero:
            continue
        fig = plt.figure(figsize=(6.0, 6.4))
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.86], projection=ccrs.PlateCarree())
        R.add_basemap(ax, labels=(day == 1))
        top = 0
        for f in sorted(ero[day], key=lambda f: int(f["properties"].get("dn") or 0)):
            dn = int(f["properties"].get("dn") or 0)
            if dn not in ERO_COLORS:
                continue
            try:
                ax.add_geometries([shape(f["geometry"])], crs=ccrs.PlateCarree(), facecolor=ERO_COLORS[dn],
                                  edgecolor="#222222", linewidth=0.8, alpha=0.55, zorder=2)
                top = max(top, dn)
            except Exception:
                pass
        m = _ero_meta(ero[day])
        fig.text(0.03, 0.955, f"ERO Day {day}", fontsize=15, fontweight="bold")
        fig.text(0.03, 0.915, m.get("valid_time") or "No risk areas issued", fontsize=10, color=R.MUTED)
        fname = f"ero_d{day}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        R.quantize_png(os.path.join(out_dir, fname))
        plt.close(fig)
        out[str(day)] = fname
    return out or None


# --------------------------------------------------------------------------- satellite
SAT_ROOT = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/SECTOR/{sector}/{band}/"
SAT_BANDS = {"GEOCOLOR": "GeoColor (visible by day, IR at night)", "13": "Infrared (band 13)"}
SAT_FRAMES = 24                     # newest frames kept per band
SAT_STEP = 2                        # keep every 2nd scan (sector scans are ~5 min, so ~10 min per frame, ~4 h loop)
SAT_SIZE = "1200x1200"


def satellite_block(sector):
    """Frame URLs for a GOES-East loop, read from the NOAA STAR CDN directory listing.

    The browser hot-links the JPEGs, so nothing is downloaded here except two small index pages.
    If a listing can't be read, the band falls back to STAR's latest.jpg (a still image)."""
    out = {"sector": sector, "bands": {}, "page": f"https://www.star.nesdis.noaa.gov/GOES/sector.php?sat=G19&sector={sector}"}
    for band, label in SAT_BANDS.items():
        root = SAT_ROOT.format(sector=sector, band=band)
        frames = []
        try:
            r = _get(root)
            r.raise_for_status()
            pat = rf'href="((\d{{11}})_GOES19-ABI-{sector}-{band}-{SAT_SIZE}\.jpg)"'
            files = sorted(set(re.findall(pat, r.text)), key=lambda x: x[1])
            for fname, stamp in files[::-1][::SAT_STEP][:SAT_FRAMES][::-1]:
                t = datetime.strptime(stamp, "%Y%j%H%M").replace(tzinfo=timezone.utc)
                frames.append({"url": root + fname, "time": t.isoformat()})
        except Exception as e:
            print(f"[sat] {band} listing failed: {e}")
        if not frames:
            frames = [{"url": root + "latest.jpg", "time": None}]
        out["bands"][band] = {"label": label, "frames": frames}
        print(f"[sat] {sector} {band}: {len(frames)} frame(s)")
    return out


# --------------------------------------------------------------------------- AFD
def fetch_afd(wfo, region="62"):
    try:
        r = _get(AFD_API.format(wfo=wfo), headers={"Accept": "application/ld+json"})
        r.raise_for_status()
        latest = r.json()["@graph"][0]
        p = _get(latest["@id"], headers={"Accept": "application/ld+json"})
        p.raise_for_status()
        return {"text": p.json()["productText"], "issued": latest.get("issuanceTime"), "source": "api.weather.gov"}
    except Exception as e:
        print(f"[afd] api.weather.gov failed ({e}); trying tgftp")
    for t in AFD_TGFTP:
        try:
            r = _get(t.format(reg=region, w=wfo.lower()))
            r.raise_for_status()
            return {"text": r.text, "issued": None, "source": t.split("/")[2]}
        except Exception as e:
            print(f"[afd] {t.split('/')[2]} failed: {e}")
    return None


def _section(text, name):
    m = re.search(rf"^\.{name}\.\.\.[^\n]*\n(.*?)(?=^&&|^\.[A-Z][A-Z /]+\.\.\.)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def _strip_issued(block):
    return re.sub(r"^Issued at [^\n]*\n", "", block.strip()).strip()


def parse_afd(text, wfo):
    head = re.search(r"Area Forecast Discussion\s*\n(National Weather Service[^\n]*)\n([^\n]+)", text)
    km = _strip_issued(_section(text, "KEY MESSAGES"))
    bullets = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*-\s+", "\n" + km) if b.strip()]
    hydro = re.sub(r"\s+", " ", _strip_issued(_section(text, "HYDROLOGY")))
    wwa = re.search(rf"\.{wfo.upper()} WATCHES/WARNINGS/ADVISORIES\.\.\.\s*\n(.*?)(?=^&&|\Z)", text, re.S | re.M)
    hazards = []
    if wwa:
        for m in re.finditer(r"^([A-Z]{2})\.\.\.(.*?)(?=^[A-Z]{2}\.\.\.|\Z)", wwa.group(1), re.S | re.M):
            body = re.sub(r"\s+", " ", m.group(2)).strip()
            if body and not body.lower().startswith("none"):
                for item in re.split(r"(?<=\.)\s+(?=[A-Z])", body):
                    if item.strip():
                        hazards.append({"area": m.group(1), "text": item.strip()})
    return {"office": head.group(1).strip() if head else f"NWS {wfo}", "issued_text": head.group(2).strip() if head else "",
            "key_messages": bullets, "hydrology": hydro[:1500], "hazards": hazards}


# --------------------------------------------------------------------------- actions
def _short(name):
    return name.split(" (")[0]


def build_actions(afd, ero, ppp_sites, meta):
    acts = []

    def add(sev, text, source, why=""):
        acts.append({"severity": sev, "text": text, "source": source, "why": why})

    if afd:
        for h in afd["hazards"]:
            add(3, f"Active NWS product ({h['area']}): {h['text']}", "AFD watches/warnings/advisories")
        for rx, sev, action in AFD_RULES:
            src = next((b for b in afd["key_messages"] if re.search(rx, b, re.I)), None)
            where = "AFD key messages"
            if not src and afd["hydrology"] and re.search(rx, afd["hydrology"], re.I):
                src, where = afd["hydrology"], "AFD hydrology"
            if src:
                add(sev, action, where, src[:260] + ("…" if len(src) > 260 else ""))
    if ero:
        for day in ("1", "2", "3"):
            hit = {n: d.get(day, 0) for n, d in ero["sites"].items() if d.get(day, 0) >= 1}
            if not hit:
                continue
            top = max(hit.values())
            names = [_short(n) for n, v in hit.items() if v == top]
            where = ", ".join(names) if len(names) <= 4 else f"{len(names)} of {len(ero['sites'])} sites, including {', '.join(names[:3])}"
            sev = 3 if top >= 3 else 2 if top == 2 else 1
            verb = {1: "Awareness", 2: "Prepare", 3: "Act", 4: "Act"}[top]
            add(sev, f"{verb}: WPC Day {day} {ERO_NAMES[top]} risk ({ERO_PCT[top]} chance of flash flooding within 25 mi) at {where}.",
                "WPC Excessive Rainfall Outlook", ero["days"].get(day, {}).get("valid_time") or "")
    if ppp_sites:
        # one item for the wettest WPC day (highest high-end anywhere on the board) and one for the 72-h total
        def worst(keys):
            c = [(n, k, v[k]) for n, v in ppp_sites.items() for k in keys if (v.get(k) or {}).get("high") is not None]
            return max(c, key=lambda x: x[2]["high"], default=None)
        for keys in (("d1", "d2", "d3"), ("72",)):
            w = worst(keys)
            if w and w[2]["high"] >= 2.0:
                n, acc, v = w
                sev = 3 if v["high"] >= 4 else 2
                when = f"Day {acc[1]}" if acc.startswith("d") else f"{acc}-h"
                add(sev, f"Plan for a high-end {when} total of {v['high']:.1f}″ at {_short(n)} "
                         f"(1-in-10 chance of more; expected {v.get('expected') or 0:.1f}″, low end {v.get('low') or 0:.1f}″).",
                    "NWS Probabilistic Precipitation Portal")
    for s in [x for x in meta.get("sites", []) if x["name"] not in R.BOARD_EXCLUDE]:
        f = s.get("ffg") or {}
        w = max(f.items(), key=lambda kv: kv[1]["ratio"], default=None)
        if w and w[1]["ratio"] >= 1.0:
            add(2, f"REFS LPMM reaches {round(w[1]['ratio'] * 100)}% of {w[0]}-h flash flood guidance at {_short(s['name'])}.",
                "REFS LPMM vs FFG", f"window ending {w[1]['ends']}")
    fl = {n: v for n, v in ((meta.get("flash") or {}).get("sites") or {}).items() if n not in R.BOARD_EXCLUDE}
    for n, v in fl.items():
        q, r = v.get("CREST_MAXUNITSTREAMFLOW") or 0, v.get("QPE_FFGMAX") or 0
        if q >= 2 or r >= 100:
            add(3 if q >= 6 else 2, f"Observed now near {_short(n)}: unit streamflow {q:.1f} m³ s⁻¹ km⁻², QPE {round(r)}% of FFG.",
                "MRMS FLASH (observed)")
    seen, uniq = set(), []
    for a in sorted(acts, key=lambda a: -a["severity"]):
        if a["text"] not in seen:
            seen.add(a["text"])
            uniq.append(a)
    if not uniq:
        uniq.append({"severity": 0, "text": "No flood-related triggers met. Routine monitoring.", "source": "All sources", "why": ""})
    return uniq


def ai_summary(afd_text, board):
    """Optional: a short AI-drafted action summary. Only runs with ANTHROPIC_API_KEY set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not afd_text:
        return None
    facts = json.dumps({k: board.get(k) for k in ("ero", "ppp_sites_primary", "actions")}, default=str)[:6000]
    prompt = ("You are supporting an emergency operations center. Using ONLY the NWS Area Forecast Discussion and the "
              "facts below, write 3 to 6 short, specific action items for the next 72 hours, most urgent first. Name "
              "times, places and amounts only when they appear in the sources. Do not invent hazards. Plain text, one "
              "item per line, no numbering.\n\nFACTS:\n" + facts + "\n\nAFD:\n" + afd_text[:12000])
    try:
        r = R.SESSION.post("https://api.anthropic.com/v1/messages", timeout=60,
                           headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                           json={"model": ANTHROPIC_MODEL, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        txt = "".join(b.get("text", "") for b in r.json().get("content", []))
        return [l.strip("-• ").strip() for l in txt.splitlines() if l.strip()][:6]
    except Exception as e:
        print(f"[board] AI summary skipped: {e}")
        return None


# --------------------------------------------------------------------------- build
def build_board(out_dir, meta, wfo, afd_region="62"):
    """Fetch everything, draw ero.png / ppp_*.png, return the meta['board'] block."""
    board = {"built": R.utcnow().isoformat(timespec="seconds"), "wfo": wfo, "images": {}, "nexrad": R.BOARD_NEXRAD}
    ppp = R.optional("ppp fetch", fetch_ppp)
    ppp_sites = None
    if ppp:
        ppp_sites = ppp_site_values(ppp)
        board["ppp"] = {"cycle": ppp["cycle"],
                        "windows": {k: {"start": v["start"], "end": v["end"], "label": ppp_label(k, v),
                                        "past_h": v.get("past_h", 0), "aligned": bool(v.get("day") or v.get("days"))}
                                    for k, v in ppp["accums"].items()},
                        "order": [k for k in PPP_ORDER if k in ppp["accums"]]}
        board["ppp_sites"] = ppp_sites
        for key in board["ppp"]["order"]:
            got = R.optional(f"ppp plot {key}", plot_ppp, ppp, key, out_dir)
            if got:
                board["images"][f"ppp_{key}"] = got[0]
                board.setdefault("ppp_panels", {})[key] = got[1]
        board["site_ll"] = {n: [round(la, 4), round(lo, 4)] for n, (la, lo, _) in board_sites().items()}
    ero = R.optional("ero fetch", fetch_ero) or {}
    ero_s = R.optional("ero summary", ero_summary, ero) if ero else None
    if ero_s:
        board["ero"] = ero_s
        maps = R.optional("ero plot", plot_ero, ero, out_dir) or {}
        for d, f in maps.items():
            board["images"][f"ero_d{d}"] = f
    afd_raw = R.optional("afd fetch", fetch_afd, wfo, afd_region)
    afd = R.optional("afd parse", parse_afd, afd_raw["text"], wfo) if afd_raw else None
    if afd:
        board["afd"] = dict(afd, source=afd_raw["source"], issued=afd_raw.get("issued"),
                            link=f"https://forecast.weather.gov/product.php?site={wfo}&issuedby={wfo}&product=AFD")
    board["actions"] = build_actions(afd, ero_s, ppp_sites, meta)
    board["sat"] = R.optional("satellite", satellite_block, R.BOARD_SAT_SECTOR)
    prim = meta.get("primary_site") or next(iter(R.SITES))
    board["ppp_sites_primary"] = (ppp_sites or {}).get(prim)
    ai = ai_summary(afd_raw["text"] if afd_raw else None, board)
    if ai:
        board["ai_actions"] = {"items": ai, "model": ANTHROPIC_MODEL}
    print(f"[board] PPP {'ok' if ppp else 'missing'}, ERO days {sorted(ero)}, AFD {'ok' if afd else 'missing'}, "
          f"{len(board['actions'])} action(s){', AI summary' if ai else ''}")
    return board
