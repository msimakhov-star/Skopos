/* Skopos — Reactor browser client.
 *
 * Opens ONE LingBot World 2 session per page, anchored on a kitchen photo,
 * and keeps it in step with the Skopos loop: every time the server's
 * render.prompt changes (lighting, occlusion, clutter — the promptable axes),
 * the new prompt is hot-swapped at the next chunk boundary.
 *
 * Verified against @reactor-models/lingbot-world-2@1.0.1 on 12 Sep 2026:
 *   new LingbotWorld2Model({})           modelName is preset by the class
 *   .on("statusChanged" | "trackReceived" | "error" | "message", fn)
 *   .connect(jwt)                         disconnected→connecting→waiting→ready
 *   .uploadFile(file) -> FileRef          then .setImage({ image: ref })
 *   .setPrompt({ prompt })  .setSeed({ seed })  .start()  .reset()
 *   .setMoveLongitudinal({ move_longitudinal: "idle"|"forward"|"back" })
 *   .setMoveLateral({ move_lateral: "idle"|"strafe_left"|"strafe_right" })
 *   .setLookHorizontal({ look_horizontal: "idle"|"left"|"right" })
 *   .setLookVertical({ look_vertical: "idle"|"up"|"down" })
 *
 * Two rules from the docs that shape this file:
 *   - setImage during generation is a silent no-op; the anchor is locked once
 *     started. Changing the anchor means reset() → setImage → start().
 *   - Movement/look values PERSIST until you send "idle". Every keydown needs
 *     its keyup, or the world drifts forever.
 *
 * Money: the session bills from ready until disconnect, $0.0070/sec, even when
 * idle. The kill timer below is not optional.
 */

const SDK_URL = "https://cdn.jsdelivr.net/npm/@reactor-models/lingbot-world-2@1.0.1/+esm";
const DEFAULT_ANCHOR = "/static/fixtures/IMG_6978.jpg";

const state = {
  model: null,
  status: "disconnected",
  jwt: null,
  anchorName: null,
  lastPrompt: null,
  started: false,
  idleKillSeconds: 90,
  idleTimer: null,
  idleDeadline: 0,
  countdownTimer: null,
  held: { lon: "idle", lat: "idle", yaw: "idle", pitch: "idle" },
};

const $ = (s) => document.querySelector(s);
const note = (t) => { const n = $("#note"); if (n) n.textContent = t; };
const badge = (t) => { const b = $("#rx-status"); if (b) b.textContent = t; };

// ---------------------------------------------------------------- idle kill
function armIdleKill(seconds) {
  clearTimeout(state.idleTimer);
  clearInterval(state.countdownTimer);
  state.idleDeadline = Date.now() + seconds * 1000;
  state.idleTimer = setTimeout(() => {
    note("idle " + seconds + "s — session closed to stop billing");
    disconnect();
  }, seconds * 1000);
  state.countdownTimer = setInterval(() => {
    const left = Math.max(0, Math.ceil((state.idleDeadline - Date.now()) / 1000));
    const el = $("#rx-idle"); if (el) el.textContent = left + "s";
  }, 500);
}
const touch = () => { if (state.model && state.status === "ready") armIdleKill(state.idleKillSeconds); };

// ---------------------------------------------------------------- lifecycle
async function loadSdk() {
  if (state.sdk) return state.sdk;
  state.sdk = await import(SDK_URL);
  return state.sdk;
}

async function connect(detail) {
  if (state.model) return state.model;
  const { LingbotWorld2Model } = await loadSdk();
  const video = $("#video");
  const m = new LingbotWorld2Model({});
  state.model = m;
  state.jwt = detail.jwt;
  state.idleKillSeconds = detail.idleKillSeconds || 90;

  m.on("statusChanged", (s) => {
    state.status = typeof s === "string" ? s : (s && s.status) || String(s);
    badge(state.status);
    if (state.status === "ready") { note("Reactor session ready"); armIdleKill(state.idleKillSeconds); }
    if (state.status === "disconnected") { state.started = false; clearTimeout(state.idleTimer); clearInterval(state.countdownTimer); }
  });
  m.on("trackReceived", (name, track, stream) => {
    if (name !== "main_video") return;
    video.srcObject = stream || new MediaStream([track]);
    video.play().catch(() => {});
  });
  m.on("error", (e) => { note("reactor error: " + (e && (e.message || e.code) || e)); console.error("reactor", e); });
  m.on("message", (msg) => {
    // command_error is how a broken precondition (start before setImage) surfaces
    if (msg && msg.type === "command_error") note("reactor: " + (msg.message || msg.error || "command error"));
  });

  badge("connecting");
  await m.connect(detail.jwt, { maxAttempts: 2 });
  return m;
}

async function stageAnchor(m, detail, anchorUrl) {
  // Fetch the fixture as a Blob and upload it. This is the ONE place image
  // bytes leave the device for the renderer, and it is counted on screen.
  const r = await fetch(anchorUrl);
  const blob = await r.blob();
  const file = new File([blob], anchorUrl.split("/").pop(), { type: blob.type || "image/jpeg" });
  const ref = await m.uploadFile(file);
  if (typeof window.rendererBytes === "number") window.rendererBytes += file.size;
  await m.setImage({ image: ref });
  state.anchorName = file.name;
  return ref;
}

async function startWith(detail, anchorUrl) {
  const m = await connect(detail);
  // wait for ready (connect resolves on ready in the SDK, but be defensive)
  if (state.status !== "ready") {
    await new Promise((res) => {
      const t = setInterval(() => { if (state.status === "ready" || state.status === "error") { clearInterval(t); res(); } }, 100);
    });
    if (state.status !== "ready") return;
  }
  if (state.started) { await m.reset(); state.started = false; }
  await stageAnchor(m, detail, anchorUrl || DEFAULT_ANCHOR);
  if (detail.seed != null) await m.setSeed({ seed: detail.seed });
  await m.setPrompt({ prompt: detail.prompt });
  state.lastPrompt = detail.prompt;
  await m.start();
  state.started = true;
  note("streaming — " + state.anchorName);
}

async function disconnect() {
  const m = state.model; if (!m) return;
  try { await idleAll(); } catch (_) {}
  try { await m.disconnect(); } catch (_) {}
  state.model = null; state.started = false; state.status = "disconnected";
  badge("disconnected");
  const v = $("#video"); if (v) v.srcObject = null;
}

// ---------------------------------------------------------------- controls
async function idleAll() {
  const m = state.model; if (!m) return;
  state.held = { lon: "idle", lat: "idle", yaw: "idle", pitch: "idle" };
  await Promise.allSettled([
    m.setMoveLongitudinal({ move_longitudinal: "idle" }),
    m.setMoveLateral({ move_lateral: "idle" }),
    m.setLookHorizontal({ look_horizontal: "idle" }),
    m.setLookVertical({ look_vertical: "idle" }),
  ]);
}

const KEYS = {
  KeyW: ["lon", "forward"], KeyS: ["lon", "back"],
  KeyA: ["lat", "strafe_left"], KeyD: ["lat", "strafe_right"],
  ArrowLeft: ["yaw", "left"], ArrowRight: ["yaw", "right"],
  ArrowUp: ["pitch", "up"], ArrowDown: ["pitch", "down"],
};
const SEND = {
  lon: (m, v) => m.setMoveLongitudinal({ move_longitudinal: v }),
  lat: (m, v) => m.setMoveLateral({ move_lateral: v }),
  yaw: (m, v) => m.setLookHorizontal({ look_horizontal: v }),
  pitch: (m, v) => m.setLookVertical({ look_vertical: v }),
};

function bindKeys() {
  const isTyping = (e) => ["INPUT", "TEXTAREA"].includes((e.target && e.target.tagName) || "");
  addEventListener("keydown", (e) => {
    if (isTyping(e) || !KEYS[e.code] || !state.model || !state.started) return;
    const [axis, val] = KEYS[e.code];
    if (state.held[axis] === val) return;          // key repeat
    state.held[axis] = val; touch();
    SEND[axis](state.model, val).catch(console.error);
    e.preventDefault();
  });
  addEventListener("keyup", (e) => {
    if (!KEYS[e.code] || !state.model) return;
    const [axis] = KEYS[e.code];
    if (state.held[axis] === "idle") return;
    state.held[axis] = "idle";
    SEND[axis](state.model, "idle").catch(console.error);
  });
  // Pointer drag on the video = look. Release = idle, always.
  const v = $("#video"); if (!v) return;
  let dragging = false, lastX = 0;
  v.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; v.setPointerCapture(e.pointerId); touch(); });
  v.addEventListener("pointermove", (e) => {
    if (!dragging || !state.model || !state.started) return;
    const dx = e.clientX - lastX;
    const want = dx > 6 ? "right" : dx < -6 ? "left" : state.held.yaw;
    if (want !== state.held.yaw) { state.held.yaw = want; SEND.yaw(state.model, want).catch(console.error); lastX = e.clientX; touch(); }
  });
  const stop = () => { dragging = false; if (state.held.yaw !== "idle" && state.model) { state.held.yaw = "idle"; SEND.yaw(state.model, "idle").catch(console.error); } };
  v.addEventListener("pointerup", stop); v.addEventListener("pointercancel", stop); v.addEventListener("pointerleave", stop);
  addEventListener("blur", () => idleAll().catch(() => {}));
}

// ---------------------------------------------------------------- Skopos hook
/* Called from index.html apply(f) on every webrtc frame. Idempotent. */
async function ensure(render) {
  const d = render.detail || {};
  if (!d.jwt) { note("reactor: no jwt in render.detail"); return; }
  if (!state.model) {
    if (state.retryAt && Date.now() < state.retryAt) return;       // back off after a failure
    try { await startWith({ ...d, prompt: render.prompt }, state.anchorUrl); }
    catch (e) {
      const ra = (e && e.retry_after_ms) || 8000;
      note(`reactor ${e && e.code ? e.code : "connect failed"} — retrying in ${Math.round(ra/1000)}s`);
      console.error(e);
      try { if (state.model) await state.model.disconnect(); } catch (_) {}
      state.model = null; state.started = false; state.status = "disconnected";
      state.retryAt = Date.now() + ra;
    }
    return;
  }
  if (state.started && render.prompt && render.prompt !== state.lastPrompt) {
    state.lastPrompt = render.prompt;
    state.model.setPrompt({ prompt: render.prompt }).catch(console.error);   // next chunk boundary
  }
}

async function setAnchor(url) {
  state.anchorUrl = url;
  if (state.model && state.started) {
    const d = state.lastDetail; if (d) await startWith(d, url);
  }
}

bindKeys();
addEventListener("beforeunload", () => { if (state.model) state.model.disconnect(); });

window.SkoposReactor = {
  ensure: (render) => { state.lastDetail = { ...(render.detail || {}), prompt: render.prompt }; return ensure(render); },
  disconnect, setAnchor, idleAll,
  get status() { return state.status; },
  get started() { return state.started; },
  get anchor() { return state.anchorName; },
};
