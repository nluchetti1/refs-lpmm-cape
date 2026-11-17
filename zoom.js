/* Zoomable image viewer shared by index.html and board.html.
 *
 *   ZoomViewer.attach("css selector for images that open the viewer")
 *
 * Scroll wheel or pinch to zoom (around the pointer), drag to pan, double-click / double-tap to
 * zoom in on a spot, + / - / 0 keys, Esc or a click on the background to close.
 * While open it follows the source image, so a loop keeps playing underneath at the same zoom.
 */
(() => {
  const css = `
  #zv { position: fixed; inset: 0; z-index: 60; background: rgba(6,11,16,.96); display: none; overflow: hidden;
        touch-action: none; user-select: none; -webkit-user-select: none; }
  #zv.open { display: block; }
  #zv img { position: absolute; left: 0; top: 0; transform-origin: 0 0; max-width: none; max-height: none;
            cursor: grab; border-radius: 0; image-rendering: auto; will-change: transform; }
  #zv.dragging img { cursor: grabbing; }
  #zv .zv-bar { position: absolute; top: 10px; right: 12px; display: flex; gap: 6px; z-index: 2; }
  #zv .zv-bar button { background: #152331; color: #e6edf2; border: 1px solid #263848; border-radius: 4px;
        font: 600 15px system-ui, sans-serif; min-width: 38px; height: 34px; padding: 0 10px; cursor: pointer; }
  #zv .zv-bar button:hover { border-color: #4fd1e8; }
  #zv .zv-hint { position: absolute; left: 12px; bottom: 10px; color: #8ea3b3; font: 13px system-ui, sans-serif; z-index: 2;
        background: rgba(15,26,36,.8); padding: 4px 8px; border-radius: 4px; pointer-events: none; }
  #zv .zv-pct { color: #8ea3b3; font: 600 13px system-ui, sans-serif; align-self: center; min-width: 46px; text-align: right; }
  @media (prefers-reduced-motion: no-preference) { #zv img.anim { transition: transform .18s ease-out; } }`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  const root = document.createElement("div");
  root.id = "zv"; root.setAttribute("role", "dialog"); root.setAttribute("aria-modal", "true"); root.setAttribute("aria-label", "Zoomable image");
  root.innerHTML = `<img alt="">
    <div class="zv-bar"><span class="zv-pct"></span>
      <button data-z="out" aria-label="Zoom out">−</button><button data-z="in" aria-label="Zoom in">+</button>
      <button data-z="fit">Fit</button><button data-z="close" aria-label="Close">✕</button></div>
    <div class="zv-hint">Scroll or pinch to zoom · drag to move · double-click to zoom in · Esc to close</div>`;
  const add = () => document.body.appendChild(root);
  document.body ? add() : document.addEventListener("DOMContentLoaded", add);
  const img = root.querySelector("img"), pct = root.querySelector(".zv-pct");

  let s = 1, x = 0, y = 0, fitS = 1, src = null, obs = null;
  const MAX = 8;
  const apply = (anim) => {
    img.classList.toggle("anim", !!anim);
    img.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
    pct.textContent = Math.round(s / fitS * 100) + "%";
  };
  const clamp = () => {
    const W = innerWidth, H = innerHeight, w = img.naturalWidth * s, h = img.naturalHeight * s;
    x = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, x));
    y = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, y));
  };
  const fit = (anim) => {
    if (!img.naturalWidth) return;
    fitS = Math.min(innerWidth * 0.98 / img.naturalWidth, innerHeight * 0.96 / img.naturalHeight);
    s = fitS; clamp(); apply(anim);
  };
  const zoomAt = (k, cx, cy, anim) => {
    const ns = Math.min(Math.max(s * k, fitS), fitS * MAX);
    x = cx - (cx - x) * ns / s; y = cy - (cy - y) * ns / s; s = ns;
    clamp(); apply(anim);
  };

  function open(source) {
    src = source;
    const first = !root.classList.contains("open");
    root.classList.add("open");
    img.onload = () => { if (first || !fitS) fit(false); else { clamp(); apply(false); } };
    img.src = source.currentSrc || source.src;
    if (img.complete && img.naturalWidth) img.onload();
    if (obs) obs.disconnect();
    obs = new MutationObserver(() => { if (src && img.src !== src.src) img.src = src.src; });   // follow a playing loop
    obs.observe(source, { attributes: true, attributeFilter: ["src"] });
    document.documentElement.style.overflow = "hidden";
  }
  function close() {
    root.classList.remove("open"); src = null;
    if (obs) { obs.disconnect(); obs = null; }
    document.documentElement.style.overflow = "";
  }

  // wheel zoom around the pointer
  root.addEventListener("wheel", e => {
    if (!root.classList.contains("open")) return;
    e.preventDefault();
    zoomAt(Math.exp(-e.deltaY * (e.deltaMode ? 0.04 : 0.0012)), e.clientX, e.clientY, false);
  }, { passive: false });

  // drag + pinch with pointer events
  const pts = new Map();
  let moved = false, last = null, pinch = null, lastTap = 0;
  root.addEventListener("pointerdown", e => {
    if (e.target.closest(".zv-bar")) return;
    root.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved = false; last = { x: e.clientX, y: e.clientY };
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) };
    }
    root.classList.add("dragging");
  });
  root.addEventListener("pointermove", e => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 2 && pinch) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      zoomAt(d / pinch.d, (a.x + b.x) / 2, (a.y + b.y) / 2, false);
      pinch.d = d; moved = true;
    } else if (pts.size === 1) {
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      x += dx; y += dy; last = { x: e.clientX, y: e.clientY };
      clamp(); apply(false);
    }
  });
  const up = e => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    if (pts.size === 1) last = [...pts.values()][0];
    if (pts.size) return;
    root.classList.remove("dragging");
    if (moved || e.type === "pointercancel") return;
    const now = Date.now();
    if (now - lastTap < 320) {                         // double tap / double click: zoom in, or back out when deep
      lastTap = 0;
      s / fitS > 3.5 ? fit(true) : zoomAt(2.5, e.clientX, e.clientY, true);
      return;
    }
    lastTap = now;
    if (e.target !== img) setTimeout(() => { if (lastTap === now) close(); }, 330);   // single click on background closes
  };
  root.addEventListener("pointerup", up);
  root.addEventListener("pointercancel", up);

  root.querySelector(".zv-bar").addEventListener("click", e => {
    const z = e.target.dataset.z; if (!z) return;
    if (z === "close") close();
    else if (z === "fit") fit(true);
    else zoomAt(z === "in" ? 1.6 : 1 / 1.6, innerWidth / 2, innerHeight / 2, true);
  });
  document.addEventListener("keydown", e => {
    if (!root.classList.contains("open")) return;
    if (e.key === "Escape") { close(); e.stopPropagation(); }
    else if (e.key === "+" || e.key === "=") zoomAt(1.6, innerWidth / 2, innerHeight / 2, true);
    else if (e.key === "-" || e.key === "_") zoomAt(1 / 1.6, innerWidth / 2, innerHeight / 2, true);
    else if (e.key === "0") fit(true);
  }, true);
  addEventListener("resize", () => { if (root.classList.contains("open")) fit(false); });

  window.ZoomViewer = {
    open, close,
    attach(selector) {
      document.addEventListener("click", e => {
        const t = e.target.closest && e.target.closest(selector);
        if (t && t.tagName === "IMG" && !root.contains(t)) { e.preventDefault(); e.stopPropagation(); open(t); }
      }, true);
    },
  };
})();
