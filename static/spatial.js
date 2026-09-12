/* Skopos spatial view: orbitable point cloud + top-down occupancy map on one 2D
   canvas. Vanilla JS, no deps, no animation loop (draw on demand only, so
   prefers-reduced-motion is respected by construction).
   Data contract: GET /api/scan/map -> see skopos/perception/depth.py to_dict(). */
(() => {
  const COL = { bg: "#0B0E0D", line: "#1E2826", text: "#7C8A86", accent: "#4FD1C5",
                free: "#1F6B48", occ: "#F0604D", unk: "#2A302E", cam: "#3B82F6" };
  const KEY = "skopos.spatial.mode";
  const MODES = ["stream", "cloud", "map"];
  const D2R = Math.PI / 180;
  const $ = id => document.getElementById(id);

  let mode = "stream", data = null, cols = null, videoWas = null;
  const cam = { yaw: 0, pitch: 0, dist: 8, tx: 0, ty: 3, tz: 0.7 };

  function resetCam() {
    cam.yaw = 35 * D2R; cam.pitch = 25 * D2R;
    cam.dist = 1.3 * ((data && data.grid_m) || 6);
    if (data && data.points.length) {                     // orbit target = centroid
      let sx = 0, sy = 0, sz = 0;
      for (const p of data.points) { sx += p[0]; sy += p[1]; sz += p[2]; }
      const n = data.points.length;
      cam.tx = sx / n; cam.ty = sy / n; cam.tz = sz / n;
    }
  }

  /* ---------- canvas plumbing ---------- */
  function setup() {
    const c = $("spatial"), g = c.getContext("2d");
    const dpr = devicePixelRatio || 1;
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = c.clientWidth, H = c.clientHeight;
    g.fillStyle = COL.bg; g.fillRect(0, 0, W, H);
    g.font = "10px ui-monospace,Menlo,monospace"; g.textBaseline = "alphabetic";
    return { g, W, H };
  }
  function message(t) {
    const { g, W, H } = setup();
    g.fillStyle = COL.text; g.textAlign = "center"; g.font = "12px ui-monospace,Menlo,monospace";
    g.fillText(t, W / 2, H / 2);
  }
  // bottom overlays (#perturb chips) sit on top of the stage; keep our text above them
  function lift() { const p = $("perturb"); return (p ? p.offsetHeight : 0) + 24; }

  /* ---------- cloud ---------- */
  // ---- route (A* on the server, drawn here) ----
  let route = null;                        // {path:[[gx,gy],...], length_m, cells_unknown, blocked, goal}
  let mapGeom = null;                      // {cs, ox, oy, rows} of the last drawMap
  async function planTo(gx, gy) {
    try {
      const r = await fetch(`/api/scan/route?gx=${gx}&gy=${gy}`);
      route = r.ok ? await r.json() : null;
    } catch (_) { route = null; }
    draw();
  }
  function drawRoute(g) {
    if (!route || !mapGeom) return;
    const { cs, ox, oy, rows } = mapGeom;
    const px = ([x, y]) => [ox + (x + 0.5) * cs, oy + (rows - 1 - y + 0.5) * cs];
    if (!route.blocked && route.path.length > 1) {
      g.beginPath();
      route.path.forEach((p, i) => { const [x, y] = px(p); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.strokeStyle = COL.accent; g.lineWidth = Math.max(2, cs * 0.45); g.lineJoin = "round"; g.lineCap = "round";
      g.globalAlpha = 0.9; g.stroke(); g.globalAlpha = 1;
    }
    const [gx, gy] = px(route.goal);
    g.strokeStyle = route.blocked ? COL.occ : COL.accent; g.lineWidth = 2;
    g.beginPath(); g.arc(gx, gy, Math.max(4, cs * 0.6), 0, Math.PI * 2); g.stroke();
    g.fillStyle = COL.text; g.font = "11px ui-monospace, monospace"; g.textAlign = "left";
    const label = route.blocked
      ? "route: blocked — goal is inside an obstacle"
      : `route ${route.length_m} m · ${route.path.length} cells · ${route.cells_unknown} unknown crossed`;
    g.fillText(label, ox, oy - 6);
  }

  function drawCloud() {
    const { g, W, H } = setup();
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const f = 0.9 * H, d = cam.dist, tx = cam.tx, ty = cam.ty, tz = cam.tz;
    // world (x right, y forward, z up) -> orbit camera -> screen; null when behind the eye
    const proj = (x, y, z) => {
      x -= tx; y -= ty; z -= tz;
      const x1 = x * cy + y * sy, y1 = -x * sy + y * cy;
      const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
      const depth = y2 + d;
      if (depth < 0.05) return null;
      const k = f / depth;
      return [W / 2 + x1 * k, H / 2 - z2 * k, depth];
    };
    const seg = (a, b) => { const p = proj(...a), q = proj(...b); if (!p || !q) return;
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); };

    // floor grid, 1 m squares over the grid extent, z = 0
    const G = data.grid_m || 6, hx = G / 2;
    g.strokeStyle = COL.line; g.lineWidth = 1;
    for (let x = -hx; x <= hx + 1e-9; x += 1) seg([x, 0, 0], [x, G, 0]);
    for (let y = 0; y <= G + 1e-9; y += 1) seg([-hx, y, 0], [hx, y, 0]);

    // points, painter's order (far first)
    const P = data.points, n = P.length;
    const SX = new Float32Array(n), SY = new Float32Array(n), DP = new Float32Array(n);
    const idx = new Uint32Array(n); let m = 0;
    for (let i = 0; i < n; i++) {
      const r = proj(P[i][0], P[i][1], P[i][2]); if (!r) continue;
      SX[i] = r[0]; SY[i] = r[1]; DP[i] = r[2]; idx[m++] = i;
    }
    const order = idx.subarray(0, m).sort((a, b) => DP[b] - DP[a]);
    for (let j = 0; j < m; j++) {
      const i = order[j], s = Math.min(4, Math.max(1.5, 2 * d / DP[i]));
      g.fillStyle = cols[i]; g.fillRect(SX[i] - s / 2, SY[i] - s / 2, s, s);
    }

    // camera marker: eye at (0,0,h) with a stem down to the floor origin
    const h = data.camera_height_m || 1.4, eye = proj(0, 0, h), foot = proj(0, 0, 0);
    if (eye && foot) {
      g.strokeStyle = COL.accent; g.lineWidth = 1; seg([0, 0, 0], [0, 0, h]);
      g.fillStyle = COL.accent; g.beginPath(); g.arc(eye[0], eye[1], 4, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(foot[0], foot[1], 2, 0, Math.PI * 2); g.fill();
      g.fillStyle = COL.text; g.textAlign = "left"; g.fillText("camera", eye[0] + 8, eye[1] + 3);
    }
    g.fillStyle = COL.text; g.textAlign = "left";
    g.fillText("drag orbit · wheel zoom · double-click reset", 12, H - lift());
  }

  /* ---------- map ---------- */
  function drawMap() {
    const { g, W, H } = setup();
    const occ = data.occupancy, rows = occ.length, colsN = occ[0].length;
    const lf = lift(), pad = 24, side = Math.max(10, Math.min(W - pad * 2, H - pad - lf));
    const cs = side / rows, ox = (W - side) / 2, oy = pad + (H - pad - lf - side) / 2;
    mapGeom = { cs, ox, oy, rows };
    const fill = { 0: COL.free, 1: COL.occ, 2: COL.unk };
    for (let r = 0; r < rows; r++) {                 // row 0 nearest the camera -> bottom
      const y = oy + (rows - 1 - r) * cs;
      for (let c = 0; c < colsN; c++) { g.fillStyle = fill[occ[r][c]] || COL.unk; g.fillRect(ox + c * cs, y, cs + 0.5, cs + 0.5); }
    }
    // 1 m grid lines
    const per = Math.max(1, Math.round(1 / (data.cell_m || 0.1)));
    g.strokeStyle = COL.line; g.lineWidth = 1;
    for (let i = 0; i <= colsN; i += per) { const x = ox + i * cs; g.beginPath(); g.moveTo(x, oy); g.lineTo(x, oy + side); g.stroke(); }
    for (let i = 0; i <= rows; i += per) { const y = oy + side - i * cs; g.beginPath(); g.moveTo(ox, y); g.lineTo(ox + side, y); g.stroke(); }
    // camera: bottom-centre for a single forward view; grid centre for a fused
    // sweep (data.centred), where the photos surround the spot you stood on.
    const centred = !!data.centred;
    const cx = ox + side / 2, cy = centred ? oy + side / 2 : oy + side;
    g.fillStyle = COL.cam; g.beginPath(); g.arc(cx, cy, 5, 0, Math.PI * 2); g.fill();
    g.fillStyle = COL.text; g.textAlign = "center";
    g.fillText(centred ? "camera at centre · first photo faces up" : "camera · forward is up", ox + side / 2, oy + side + 14);
    // legend, top-right, below the header badges
    const lx = W - 118, ly = 64;
    [["free", COL.free], ["occupied", COL.occ], ["unknown", COL.unk]].forEach(([t, c], i) => {
      g.fillStyle = c; g.fillRect(lx, ly + i * 16, 10, 10);
      g.fillStyle = COL.text; g.textAlign = "left"; g.fillText(t, lx + 16, ly + i * 16 + 9);
    });
    g.fillStyle = COL.text; g.textAlign = "left";
    g.fillText(`depth ${Number(data.depth_ms).toFixed(1)} ms · ${data.n_points} points`, 12, H - lf - 14);
    g.fillText(data.note || "", 12, H - lf);
    drawRoute(g);
  }

  function draw() {
    if (mode === "stream") return;
    const c = $("spatial"); if (!c) return;
    c.style.cursor = mode === "cloud" ? "grab" : (mode === "map" ? "crosshair" : "default");
    if (!data) return message("no scan yet — press scan photo");
    mode === "cloud" ? drawCloud() : drawMap();
  }
  let pending = false;   // coalesce pointermove bursts into one draw per frame
  const requestDraw = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; draw(); }); };

  /* ---------- public ---------- */
  async function load() {
    route = null;
    try {
      const r = await fetch("/api/scan/map");
      if (!r.ok) { data = cols = null; draw(); return; }
      data = await r.json();
      cols = data.points.map(p => `rgb(${p[3]},${p[4]},${p[5]})`);
      resetCam();
    } catch (e) { data = cols = null; console.warn("spatial: load failed", e); }
    draw();
  }

  function setMode(m) {
    if (!MODES.includes(m)) m = "stream";
    mode = m;
    try { localStorage.setItem(KEY, m); } catch (e) {}
    const v = $("video"), c = $("spatial");
    if (m === "stream") {
      c.style.display = "none";
      if (v && videoWas !== null) { v.style.display = videoWas; videoWas = null; }
    } else {
      if (v && videoWas === null) { videoWas = v.style.display; v.style.display = "none"; }
      c.style.display = "block";
    }
    document.querySelectorAll("[data-spatial]").forEach(b => {
      const on = b.dataset.spatial === m;
      b.setAttribute("aria-pressed", on);
      b.style.color = on ? COL.accent : ""; b.style.borderColor = on ? COL.accent : "";
    });
    draw();
  }

  window.SkoposSpatial = { load, setMode, get mode() { return mode; } };

  /* ---------- wiring ---------- */
  const c = $("spatial");
  if (c) {
    c.style.touchAction = "none";
    let drag = null;
    c.addEventListener("pointerdown", e => {
      if (mode === "map" && mapGeom && data) {
        const rect = c.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const { cs, ox, oy, rows } = mapGeom;
        const gx = Math.floor((x - ox) / cs), gy = rows - 1 - Math.floor((y - oy) / cs);
        if (gx >= 0 && gx < rows && gy >= 0 && gy < rows) planTo(gx, gy);
        return;
      }
      if (mode !== "cloud") return; drag = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); c.style.cursor = "grabbing";
    });
    c.addEventListener("pointermove", e => {
      if (!drag) return;
      const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY];
      cam.yaw -= dx * 0.006;
      cam.pitch = Math.max(-80 * D2R, Math.min(80 * D2R, cam.pitch + dy * 0.006));
      requestDraw();
    });
    const up = () => { drag = null; c.style.cursor = mode === "cloud" ? "grab" : "default"; };
    c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
    c.addEventListener("wheel", e => { if (mode !== "cloud") return; e.preventDefault();
      cam.dist = Math.max(0.5, Math.min(40, cam.dist * Math.exp(e.deltaY * 0.0015))); requestDraw(); }, { passive: false });
    c.addEventListener("dblclick", () => { if (mode !== "cloud") return; resetCam(); draw(); });
  }
  document.querySelectorAll("[data-spatial]").forEach(b => b.addEventListener("click", () => setMode(b.dataset.spatial)));
  addEventListener("resize", () => draw());
  addEventListener("skopos:scanned", () => load());

  let saved = "stream";
  try { saved = localStorage.getItem(KEY) || "stream"; } catch (e) {}
  setMode(saved);
  load();
})();
