// Every piece of text that FLOATS IN THE 3-D SCENE must obey the 🏷 labels
// toggle. The bug this exists for: the flag's only reader was `Duck.tsx`, and
// a non-duck body draws as `SimStage.RobotBody`, which has no name label — so
// on a moss-yard the button governed nothing at all, while `SimCamView`'s
// "MOSS sees can1 · 0.1 m" tags (the only labels that page has) stayed on
// screen under the sensor-overlay toggle instead.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { whyLabel } from "./SimCamView";

/** A `why` row as the stream sends it (lib/sim `DetPayload.why`). */
const SEEN = ["can1", "seen", 1, 0.4, 0.1, 0.05, 0.03] as unknown as Parameters<typeof whyLabel>[0];
const SMALL = ["cap0", "small", 0, 1.2, 0.3, 0.02, 0.01, 0.8] as unknown as Parameters<typeof whyLabel>[0];

describe("the 🏷 toggle decides whether a scene label has any text", () => {
  it("produces the label when it is on", () => {
    expect(whyLabel(SEEN, 0.42, true)).toBe("MOSS sees can1 · 0.4 m");
    expect(whyLabel(SMALL, 1.7, true)).toContain("can't see cap0");
  });

  it("produces NOTHING when it is off — for every kind of row", () => {
    for (const row of [SEEN, SMALL]) expect(whyLabel(row, 1.0, false)).toBeNull();
  });

  it("asks for the flag rather than trusting the caller to check first", () => {
    // The gate is a REQUIRED argument, so a new caller that forgets does not
    // compile. Guarding it here too, because deleting the parameter would
    // silently restore the old signature for JS callers.
    expect(whyLabel.length).toBe(3);
  });
});

// ---------------------------------------------------------------- structural
// Weaker than the behaviour above and known to be: it catches a component that
// anchors text to the scene and never mentions the flag AT ALL. It cannot
// catch one that mentions it and then ignores it — A/B'd against exactly that
// and it passed, which is why the real guard is the signature above.
const DIR = join(__dirname);

/** `<Html center>` / `<Html position=…>` anchors content to a point in the
 *  scene — a floating label. `<Html fullscreen>` does not: SimCamView and
 *  SimMossMap use it only to portal a docked PANEL to <body>, and a panel has
 *  its own collapse control. */
const ANCHORED = /<Html\s+(center|position)/;

/** Anchors text but is not the /sim scene, with the reason. An entry here is
 *  a decision, not an exemption to reach for. */
const NOT_SIM: Record<string, string> = {
  "PoseDuck.tsx": "the 🎬 pose editor's joint handles — a different page, and " +
    "they are the thing you are dragging rather than an annotation of it",
};

function sceneLabelFiles(): string[] {
  return readdirSync(DIR)
    .filter((f) => f.endsWith(".tsx"))
    .filter((f) => ANCHORED.test(readFileSync(join(DIR, f), "utf8")));
}

describe("every component that anchors text to the scene knows about the flag", () => {
  it("finds them at all", () => {
    // If this drops to nothing the regex has rotted and the next test would
    // pass vacuously.
    const files = sceneLabelFiles();
    expect(files).toContain("Duck.tsx");
    expect(files).toContain("SimCamView.tsx");
  });

  it("each one names getDuckLabels, or says why it is exempt", () => {
    for (const f of sceneLabelFiles()) {
      if (f in NOT_SIM) continue;
      const src = readFileSync(join(DIR, f), "utf8");
      expect(src, `${f} anchors text to the 3-D scene but never mentions ` +
        `getDuckLabels — the 🏷 button would not turn it off. Gate it, or ` +
        `add it to NOT_SIM with the reason.`).toContain("getDuckLabels");
    }
  });
});
