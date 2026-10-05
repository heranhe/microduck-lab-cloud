// film.mjs — headless Chrome over CDP, Page.startScreencast -> numbered jpegs
// plus an ffconcat manifest carrying each frame's REAL duration, so a stream
// that arrives irregularly still retimes to a constant fps without drifting.
//   node film.mjs --url http://localhost:63317/sim --out /tmp/film --wait 8000 \
//        --skip 37000 --seconds 21 --keys "w@600,w@600"
// Node 22's global WebSocket drives CDP; no dependencies. Shape borrowed from
// captures/showcase-2026-09-17/tools/shoot.mjs, which takes stills.
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME ||
  `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`;
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : process.argv[i + 1]; };
const pageUrl = arg("url"), out = arg("out", "/tmp/film");
const W = +arg("w", 1600), H = +arg("h", 900);
const wait = +arg("wait", 8000);         // page load + first frames
const skipMs = +arg("skip", 0);          // let the world run this long first
const seconds = +arg("seconds", 20);     // then film this long
const port = +arg("port", 9444);
const reset = arg("reset", "");          // scenario to POST to /world/load before filming
const lab = arg("lab", "http://127.0.0.1:8788");
const keys = (arg("keys", "") || "").split(",").filter(Boolean)
  .map((k) => { const [key, ms] = k.split("@"); return { key, ms: +(ms || 400) }; });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
setTimeout(() => { console.error("watchdog: giving up"); try { chrome.kill(); } catch {} process.exit(3); },
  +(process.env.FILM_MAX_MS || 600000)).unref();

mkdirSync(out, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), "film-"));
const chrome = spawn(CHROME, [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  ...(process.env.CHROME ? [] : ["--headless=new"]),
  "--hide-scrollbars", "--mute-audio", "--enable-gpu", "--use-angle=metal",
  "--ignore-gpu-blocklist", `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });
process.on("exit", () => chrome.kill());

let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch {}
}
if (!target) { console.error("chrome did not come up"); process.exit(1); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let seq = 0; const pending = new Map(); const errors = [];
const frames = [];                       // {t, buf}
let filming = false;
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
  if (d.method === "Runtime.exceptionThrown")
    errors.push("pageerror: " + (d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text));
  if (d.method === "Page.screencastFrame") {
    // ACK every frame or Chrome stops sending after a few.
    send("Page.screencastFrameAck", { sessionId: d.params.sessionId });
    if (filming) frames.push({ t: d.params.metadata.timestamp, b64: d.params.data });
  }
};
const send = (method, params = {}) => new Promise((res) => { const id = ++seq; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });

await send("Page.enable"); await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: pageUrl });
await sleep(wait);

for (const { key, ms } of keys) {
  await send("Input.dispatchKeyEvent", { type: "keyDown", key, text: key.length === 1 ? key : undefined });
  await sleep(ms);                      // HELD for ms — camera flight needs a hold, not a tap
  await send("Input.dispatchKeyEvent", { type: "keyUp", key });
  await sleep(120);
}

// Reset the world AFTER the page is up and the camera is placed, so the
// filmed window starts at a known sim time: `--skip` is then measured from
// t = 0 and the scout run's timings (record-world, same seed) still apply.
if (reset) {
  const r = await fetch(`${lab}/world/load`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario: reset }),
  });
  if (!r.ok) { console.error(`world/load ${r.status}: ${await r.text()}`); process.exit(4); }
  await sleep(1500);                    // compose + install, then t starts
}
if (skipMs > 0) await sleep(skipMs);
await send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
filming = true;
await sleep(seconds * 1000);
filming = false;
await send("Page.stopScreencast");

if (!frames.length) { console.error("no frames captured"); process.exit(2); }
const t0 = frames[0].t;
let manifest = "ffconcat version 1.0\n";
frames.forEach((f, i) => {
  const name = `f${String(i).padStart(5, "0")}.jpg`;
  writeFileSync(join(out, name), Buffer.from(f.b64, "base64"));
  const next = frames[i + 1]?.t ?? f.t + 0.05;
  manifest += `file ${name}\nduration ${(next - f.t).toFixed(4)}\n`;
});
manifest += `file f${String(frames.length - 1).padStart(5, "0")}.jpg\n`;
writeFileSync(join(out, "frames.ffconcat"), manifest);
console.log(`frames ${frames.length}, span ${(frames[frames.length - 1].t - t0).toFixed(2)}s, ` +
  `mean fps ${(frames.length / (frames[frames.length - 1].t - t0)).toFixed(1)}`);
if (errors.length) console.log("page errors:\n" + errors.slice(0, 5).join("\n"));
process.exit(0);
