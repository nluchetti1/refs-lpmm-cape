"""
Refresh only the MRMS FLASH panel on the published site (runs every 30 min).

Starts from data/site_cache (the last full build), redraws flash.png and the
meta.json "flash" block, then writes ./site for deployment. The 6-hourly main
build also draws a FLASH snapshot, so the panel is never older than ~30 min
while this workflow runs, and never older than 6 h if it stops.
"""
import os
import shutil
import sys
from datetime import datetime

import refs_qpf as R
import lookback as L


def main():
    R.init_sites()
    if not os.path.isdir(R.SITE_CACHE):
        print("[flash] no cached site yet; run the main workflow once first")
        sys.exit(1)
    if os.path.isdir(R.OUT_DIR):
        shutil.rmtree(R.OUT_DIR)
    shutil.copytree(R.SITE_CACHE, R.OUT_DIR)
    if os.path.exists(R.INDEX_SRC):
        shutil.copy(R.INDEX_SRC, os.path.join(R.OUT_DIR, "index.html"))
        R.copy_pages(R.OUT_DIR)
    fl = R.build_flash_now(R.OUT_DIR)
    if not fl:
        sys.exit(1)
    meta_p = os.path.join(R.OUT_DIR, "meta.json")
    meta = R.load_json(meta_p, {})
    meta["flash"] = fl
    meta.setdefault("products", {})["flash"] = True
    # Keep the situational awareness board fresh (PPP updates hourly; ERO/AFD several times a day)
    built = (meta.get("board") or {}).get("built")
    age = (R.utcnow() - datetime.fromisoformat(built)).total_seconds() / 60 if built else 1e9
    if age > R.BOARD_MAX_AGE_MIN:
        b = R.optional("board", R.build_board_block, meta)
        if b:
            meta["board"] = b
            meta["products"]["board"] = True
            # persist into the cached site so the next refresh starts from this board, not the 6-hourly one
            cmeta_p = os.path.join(R.SITE_CACHE, "meta.json")
            cmeta = R.load_json(cmeta_p, {})
            cmeta["board"] = b
            cmeta.setdefault("products", {})["board"] = True
            R.save_json(cmeta_p, cmeta, indent=1)
            for f in b.get("images", {}).values():
                shutil.copy(os.path.join(R.OUT_DIR, f), os.path.join(R.SITE_CACHE, f))
    if meta.get("board"):                       # satellite frames are cheap, so refresh them every run
        import board as B
        sat = R.optional("satellite", B.satellite_block, R.BOARD_SAT_SECTOR)
        if sat:
            meta["board"]["sat"] = sat
        qpe = R.optional("mrms qpe", B.mrms_qpe_block, R.OUT_DIR)     # new Pass 1 hour every run
        if qpe:
            meta["board"]["qpe"] = qpe
    meta["products"]["cases"] = R.publish_cases(R.OUT_DIR) > 0
    R.save_json(meta_p, meta, indent=1)
    print(f"[flash] site refreshed, FLASH valid {fl['valid']}")


if __name__ == "__main__":
    main()
