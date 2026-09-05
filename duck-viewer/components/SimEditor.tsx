"use client";

// Scenario editor for /sim: pick a tool, click the floor to place walls
// (two clicks), boxes, balls and duck spawns, remove things from the list,
// then save under a name (PUT /scenarios/{name}) and load it. Editing works
// on a DRAFT copy; the world keeps running on the loaded scenario until you
// save-and-load. Coordinates are MuJoCo world metres, z up.

import { useState } from "react";
import { LAB_HTTP } from "@/lib/lab";
import {
  loadWorld,
  ROLE_NAMES,
  TEAM_COLORWAYS,
  TEAM_NAMES,
  teamColor,
  type RoleName,
  type Scenario,
  type ScenarioDuck,
  type TeamName,
  type WorldInfo,
} from "@/lib/sim";

export type EditorTool = "wall" | "box" | "ball" | "duck" | "person" | "toy" | "basket" | null;

export interface EditorState {
  draft: Scenario;
  tool: EditorTool;
  wallStart: [number, number] | null;
}

const PANEL: React.CSSProperties = {
  position: "absolute",
  top: 56,
  left: 10,
  width: 250,
  background: "rgba(16,18,22,0.9)",
  border: "1px solid #2b313b",
  borderRadius: 6,
  color: "#e9edf1",
  fontFamily: "ui-monospace, Menlo, monospace",
  fontSize: 12,
  padding: "8px 10px",
  zIndex: 20,
  maxHeight: "60vh",
  overflowY: "auto",
};
const BTN: React.CSSProperties = {
  background: "#1f242c",
  color: "#e9edf1",
  border: "1px solid #2b313b",
  borderRadius: 4,
  padding: "2px 7px",
  fontFamily: "inherit",
  fontSize: 12,
  cursor: "pointer",
};

export function emptyDraft(base: Scenario | null): Scenario {
  if (base) return JSON.parse(JSON.stringify(base));
  return {
    version: 1,
    name: "my-room",
    seed: 0,
    floor: { size: [4, 4] },
    walls: [],
    boxes: [],
    balls: [],
    ducks: [],
    collision: "walk",
  };
}

/** Apply one floor click to the draft with the active tool. Returns the new
 *  state (a wall needs two clicks; the first only arms `wallStart`). */
export function applyFloorClick(st: EditorState, x: number, y: number): EditorState {
  const d = st.draft;
  const r = (v: number) => Math.round(v * 100) / 100;
  x = r(x);
  y = r(y);
  switch (st.tool) {
    case "wall":
      if (!st.wallStart) return { ...st, wallStart: [x, y] };
      if (Math.hypot(x - st.wallStart[0], y - st.wallStart[1]) < 0.05) return { ...st, wallStart: null };
      return {
        ...st,
        wallStart: null,
        draft: { ...d, walls: [...d.walls, { from: st.wallStart, to: [x, y], height: 0.3, thickness: 0.02 }] },
      };
    case "box":
      return {
        ...st,
        draft: { ...d, boxes: [...d.boxes, { pos: [x, y, 0.075], size: [0.2, 0.2, 0.15], yaw: 0, mass: 0, rgba: [0.55, 0.45, 0.35, 1] }] },
      };
    case "ball":
      return { ...st, draft: { ...d, balls: [...d.balls, { pos: [x, y], radius: 0.035, mass: 0.015 }] } };
    case "duck": {
      let n = d.ducks.length;
      while (d.ducks.some((k) => k.id === `d${n}`)) n++;
      // On a pitch a duck joins the team whose half it is standing in, facing
      // the goal it attacks — which is the only placement that is not an
      // immediate scenario error, and the one a person means by clicking
      // their own half.
      const pitch = (d.goal_width ?? 0) > 0;
      const home = (d.ducks.find((k) => k.spawn[0] < 0)?.team ?? "cream") as TeamName;
      const away = (d.ducks.find((k) => k.spawn[0] > 0)?.team ?? (home === "sky" ? "cream" : "sky")) as TeamName;
      const mine = x < 0 ? home : away;
      const duck: ScenarioDuck = {
        id: `d${n}`,
        spawn: [x, y, pitch ? (x < 0 ? 0 : Math.PI) : 0],
        policy: "pollen:alpha_walking",
        tof: "datasheet",
        ...(pitch ? { brain: "chase", team: mine } : {}),
      };
      return { ...st, draft: { ...d, ducks: [...d.ducks, duck] } };
    }
    case "person": {
      const persons = d.persons ?? [];
      let n = persons.length;
      while (persons.some((q) => q.id === `p${n}`)) n++;
      // A default patrol: back and forth across the floor from the click.
      return {
        ...st,
        draft: { ...d, persons: [...persons, { id: `p${n}`, pos: [x, y], yaw: 0, path: [[x, y], [-x, y]], speed: 0.25, radius: 0.2, height: 1.0 }] },
      };
    }
    case "toy": {
      const toys = d.pickables ?? [];
      let n = toys.length;
      while (toys.some((q) => q.id === `t${n}`)) n++;
      const kinds = ["brick", "block", "sock"] as const;
      return { ...st, draft: { ...d, pickables: [...toys, { id: `t${n}`, kind: kinds[n % 3], pos: [x, y], yaw: 0 }] } };
    }
    case "basket":
      return { ...st, draft: { ...d, basket: { pos: [x, y], size: [0.3, 0.3], rim: 0.06 } } };
    default:
      return st;
  }
}

export async function saveScenario(name: string, sc: Scenario): Promise<Scenario> {
  const r = await fetch(`${LAB_HTTP}/scenarios/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...sc, name }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `save ${r.status}`);
  return r.json();
}

export function SimEditor({
  state,
  setState,
  onClose,
  onLoaded,
  top,
  brains,
}: {
  state: EditorState;
  setState: (s: EditorState) => void;
  onClose: () => void;
  onLoaded: (w: WorldInfo) => void;
  /** y the page's top bar actually ends at — it wraps rows on a narrow window. */
  top?: number;
  /** Brain kinds the server has (`GET /world`), so the picker offers what
   *  actually exists rather than a list that goes stale. */
  brains?: string[];
}) {
  // Built-ins are read-only on the server, so a draft of one saves as a copy.
  const [name, setName] = useState(state.draft.name ? `${state.draft.name}-edit` : "my-room");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const d = state.draft;
  const setDraft = (draft: Scenario) => setState({ ...state, draft });
  const setDuck = (i: number, patch: Partial<ScenarioDuck>) =>
    setDraft({ ...d, ducks: d.ducks.map((q, j) => (j === i ? { ...q, ...patch } : q)) });
  const pitch = (d.goal_width ?? 0) > 0;
  const brainKinds = brains ?? ["wander", "follow", "chase", "tidy", "script"];
  /** Make it a pitch, or stop being one: goals on the short walls, and the
   *  ducks split by the half they already stand in so the scene is legal the
   *  moment it is toggled (a team facing both goals is refused on save). */
  const togglePitch = () => {
    if (pitch) {
      setDraft({ ...d, goal_width: 0, ducks: d.ducks.map((q) => ({ ...q, team: null, role: null })) });
      return;
    }
    const ducks = d.ducks.map((q) => ({
      ...q,
      brain: q.brain ?? "chase",
      team: (q.spawn[0] < 0 ? "cream" : "sky") as TeamName,
      spawn: [q.spawn[0], q.spawn[1], q.spawn[0] < 0 ? 0 : Math.PI] as [number, number, number],
    }));
    setDraft({ ...d, goal_width: 0.7, ducks, balls: d.balls.length ? d.balls : [{ pos: [0, 0], radius: 0.035, mass: 0.015 }] });
  };
  const tool = (t: EditorTool) => setState({ ...state, tool: state.tool === t ? null : t, wallStart: null });
  const toolBtn = (t: EditorTool, label: string) => (
    <button key={label} style={{ ...BTN, borderColor: state.tool === t ? "#f2b632" : "#2b313b" }} onClick={() => tool(t)}>
      {label}
    </button>
  );
  const row = (label: string, onRemove: () => void, k: string) => (
    <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 6, color: "#c9d0d8" }}>
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
      <button style={{ ...BTN, padding: "0 5px" }} onClick={onRemove} title="remove">
        ✕
      </button>
    </div>
  );
  const saveLoad = async () => {
    setBusy(true);
    try {
      await saveScenario(name, d);
      const w = await loadWorld(name);
      onLoaded(w);
      setMsg(`saved + loaded ${name}`);
      onClose();
    } catch (e) {
      setMsg(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };
  const floorInput = (i: 0 | 1) => (
    <input
      type="number"
      step={0.5}
      min={0.5}
      max={20}
      value={d.floor.size[i]}
      onChange={(e) => {
        const size: [number, number] = [...d.floor.size] as [number, number];
        size[i] = Number(e.target.value);
        setDraft({ ...d, floor: { size } });
      }}
      style={{ ...BTN, width: 56, padding: "1px 4px" }}
    />
  );
  return (
    <div style={{ ...PANEL, top: top ?? PANEL.top, maxHeight: `calc(100vh - ${(top ?? 56) + 10}px)` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>World editor</span>
        <button style={BTN} onClick={onClose}>
          close
        </button>
      </div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
        {toolBtn("wall", "▭ wall")}
        {toolBtn("box", "▣ box")}
        {toolBtn("ball", "● ball")}
        {toolBtn("duck", "🦆 duck")}
        {toolBtn("person", "🧍 person")}
        {toolBtn("toy", "🧱 toy")}
        {toolBtn("basket", "🧺 basket")}
      </div>
      <div style={{ color: "#9aa5b1", marginBottom: 6 }}>
        {state.tool === "wall"
          ? state.wallStart
            ? "click the wall's other end"
            : "click where the wall starts"
          : state.tool
            ? `click the floor to place a ${state.tool}`
            : "pick a tool, then click the floor"}
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}>
        floor {floorInput(0)} × {floorInput(1)} m
        <button
          style={{ ...BTN, borderColor: pitch ? "#43c2b8" : "#2b313b", marginLeft: "auto" }}
          onClick={togglePitch}
          title="goals on both short walls, a ball on the spot, and every duck on the team of its own half — cream at −x, sky at +x"
        >
          {pitch ? "⚽ pitch" : "make a pitch"}
        </button>
      </div>
      {d.walls.map((w, i) =>
        row(`wall ${i}: (${w.from.join(",")}) → (${w.to.join(",")})`, () => setDraft({ ...d, walls: d.walls.filter((_, k) => k !== i) }), `w${i}`)
      )}
      {d.boxes.map((b, i) =>
        row(`box ${i}: ${b.pos[0]},${b.pos[1]} · ${b.size.map((v) => v.toFixed(2)).join("×")}${b.mass ? ` · ${b.mass} kg` : ""}`, () => setDraft({ ...d, boxes: d.boxes.filter((_, k) => k !== i) }), `b${i}`)
      )}
      {d.balls.map((b, i) => row(`ball ${i}: ${b.pos[0]},${b.pos[1]}`, () => setDraft({ ...d, balls: d.balls.filter((_, k) => k !== i) }), `k${i}`))}
      {d.ducks.map((k, i) => (
        <div key={`d${i}`} style={{ color: "#c9d0d8", marginBottom: 2 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 6, alignItems: "center" }}>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "flex", gap: 5, alignItems: "center" }}>
              {k.team && (
                <span style={{ width: 9, height: 9, borderRadius: 2, background: teamColor(k.team) ?? "#555", border: "1px solid rgba(0,0,0,.4)" }} />
              )}
              {k.id}: {k.spawn[0]},{k.spawn[1]} · {k.policy?.split(":").pop() ?? "stand"}
            </span>
            <select
              value={k.brain ?? ""}
              onChange={(e) => setDuck(i, { brain: e.target.value || null })}
              style={{ ...BTN, padding: "0 3px" }}
              title="brain in auto mode"
            >
              {/* Every kind the server actually has, so `chase` (the soccer
                  brain) and the learned ones are reachable — a hard-coded
                  four could not name the brain the pitch runs. */}
              <option value="">auto</option>
              {brainKinds.map((b) => (
                <option key={b} value={b}>{b.startsWith("learned:") ? b.slice(8) : b}</option>
              ))}
            </select>
            <button style={{ ...BTN, padding: "0 5px" }} onClick={() => setDraft({ ...d, ducks: d.ducks.filter((_, j) => j !== i) })} title="remove">
              ✕
            </button>
          </div>
          {pitch && (
            <div style={{ display: "flex", gap: 4, alignItems: "center", paddingLeft: 14, marginTop: 1 }}>
              <select
                value={k.team ?? ""}
                onChange={(e) => setDuck(i, { team: (e.target.value || null) as TeamName | null, ...(e.target.value ? {} : { role: null }) })}
                style={{ ...BTN, padding: "0 3px", color: teamColor(k.team) ?? "#e9edf1" }}
                title="the team's colorway — what the duck is painted, and the only thing that tells two ducks apart on the robot"
              >
                <option value="">no team</option>
                {TEAM_NAMES.map((t) => (
                  <option key={t} value={t}>{TEAM_COLORWAYS[t].label}</option>
                ))}
              </select>
              <select
                value={k.role ?? ""}
                onChange={(e) => setDuck(i, { role: (e.target.value || null) as RoleName | null })}
                style={{ ...BTN, padding: "0 3px" }}
                disabled={!k.team}
                title="what it plays; blank = the team board picks one attacker and the rest support"
              >
                <option value="">any role</option>
                {ROLE_NAMES.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      ))}
      {(d.pickables ?? []).map((q, i) => row(`${q.id}: ${q.kind} at ${q.pos[0]},${q.pos[1]}`, () => setDraft({ ...d, pickables: (d.pickables ?? []).filter((_, j) => j !== i) }), `t${i}`))}
      {d.basket && row(`basket at ${d.basket.pos[0]},${d.basket.pos[1]} · rim ${d.basket.rim} m`, () => setDraft({ ...d, basket: null }), "basket")}
      {(d.persons ?? []).map((q, i) =>
        row(`${q.id}: ${q.pos[0]},${q.pos[1]} · ${q.path.length} waypoints · ${q.speed} m/s`, () => setDraft({ ...d, persons: (d.persons ?? []).filter((_, j) => j !== i) }), `p${i}`)
      )}
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8 }}>
        <input value={name} onChange={(e) => setName(e.target.value)} style={{ ...BTN, flex: 1, padding: "2px 6px" }} placeholder="scenario name" />
        <button style={{ ...BTN, borderColor: "#43c2b8" }} disabled={busy || !d.ducks.length} onClick={saveLoad} title="PUT /scenarios then load">
          {busy ? "…" : "save + load"}
        </button>
      </div>
      {!d.ducks.length && <div style={{ color: "#f2b632", marginTop: 4 }}>add at least one duck</div>}
      {msg && <div style={{ color: "#9aa5b1", marginTop: 4 }}>{msg}</div>}
    </div>
  );
}
