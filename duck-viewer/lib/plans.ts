import { localizeBehavior } from "./teachLocalization";
import { LAB_HTTP } from "./lab";
import type { Clip } from "./anim";
export type LocalPlan = { behavior: string; steps: number; weights: Record<string, number>; stageSteps: Record<string, number>; stageWeights: Record<string, Record<string, number>> };
export type CloudPlan = { task: string; iterations: number; envs: number; gpu: string; flavor: string };
export type Plan = { version: 1; id?: string; revision?: number; title: string; description: string; success: string; robot: string; source: string; local: LocalPlan | null; cloud: CloudPlan | null; reference?: Clip; path?: string; revisions?: number[]; storage?: string };
export type Term = { key: string; friendly: string; weight: number; isPenalty: boolean };
export type Recipe = { id: string; robot: string; emoji: string; title: string; description: string; howItLearns: string; successMetric: string; defaultSteps: number; terms: Term[]; availableTerms: Term[]; curriculum: {label: string; steps: number; detail?: string}[] };
export type Catalog = {templates: Plan[]; behaviors: Recipe[]};
export async function planRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const r = await fetch(`${LAB_HTTP}${path}`, {method, headers: body === undefined ? {} : {"Content-Type": "application/json"}, body: body === undefined ? undefined : JSON.stringify(body), cache: "no-store"});
  const data = await r.json();
  if (r.status === 404) throw new Error(`训练服务 ${LAB_HTTP} 缺少方案接口，可能仍在运行旧版本。请等待正在进行的训练完成后重启 duck-lab，再刷新页面。`);
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "请求失败，请检查配置");
  return data as T;
}
/** Remove display-only properties before saving/exporting. */
export function planFile(p: Plan): Plan {
  return {version: 1, ...(p.id && !p.id.startsWith("template-") ? {id:p.id, revision:p.revision} : {}), title:p.title, description:p.description, success:p.success, robot:p.robot, source:p.source,
    local:p.local, cloud:p.cloud, ...(p.reference ? {reference:p.reference} : {})};
}
export function newFromTemplate(p: Plan): Plan {
  const result = structuredClone(planFile(p)); delete result.id; delete result.revision;
  return result;
}
export function clipPlan(clip: Clip, recipes: Recipe[]): Plan {
  const robot = clip.robot || "microduck";
  const b = recipes.find(r => r.robot === robot && r.id === (robot === "microduck" ? "imitate" : `${robot}_imitate`));
  if (!b) throw new Error("此机器人尚未接入关键帧模仿训练");
  return {version:1, title:`模仿 · ${clip.name}`, description:`学习参考动作“${clip.name}”，并在物理环境中完成动作。`, success:b.successMetric, robot, source:"animation", reference:clip,
    local:{behavior:b.id, steps:b.defaultSteps, weights:Object.fromEntries(b.terms.map(t=>[t.key,t.weight])), stageSteps:{}, stageWeights:{}}, cloud:null};
}
export function pythonPlan(p: Plan): string {
  const literal = (v: unknown): string => {
    if (v === null || v === undefined) return "None";
    if (typeof v === "boolean") return v ? "True" : "False";
    if (Array.isArray(v)) return `[${v.map(literal).join(", ")}]`;
    if (typeof v === "object") return `{${Object.entries(v as Record<string, unknown>).map(([k,x]) => `${JSON.stringify(k)}: ${literal(x)}`).join(", ")}}`;
    return JSON.stringify(v);
  };
  return `# MicroDuck declarative training plan\nTRAINING_PLAN = ${literal(planFile(p))}\n`;
}
export function downloadJSON(value: unknown, name: string) {
  downloadText(JSON.stringify(value,null,2),name,"application/json");
}
export function downloadText(value: string, name: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([value],{type}));
  const a=document.createElement("a"); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
export function planNavigationQuery(search: string): URLSearchParams {
  const query = new URLSearchParams(search);
  for (const key of ["clip", "plan", "revision"]) query.delete(key);
  return query;
}
export function plansHref(clip?: string): string {
  const query = planNavigationQuery(typeof window !== "undefined" ? window.location.search : "");
  if (clip) query.set("clip",clip);
  return `/plans${query.size ? `?${query}` : ""}`;
}

export function localizedCatalog(c: Catalog): Catalog {
  const names: Record<string,string> = {one_leg:"单脚站立",stand:"保持站立",crouch:"下蹲",deep_squat:"深蹲",spin:"原地转圈",headstand:"头倒立",backflip:"后空翻",jump_backflip:"跳跃后空翻",airflip:"跳跃后空翻",run:"向前跑",find_ball:"寻找球",getup:"从地面起身",imitate:"关键帧动作模仿",g1_imitate:"关键帧动作模仿（G1）",g1_stand:"保持站立（G1）",g1_squat:"保持下蹲（G1）",g1_kick:"前踢（G1）",g1_punch:"直拳（G1）",kick_right:"右脚踢球",kick_left:"左脚踢球",kick_right_wide:"右脚踢球 · 宽范围",kick_left_wide:"左脚踢球 · 宽范围",mars_reach:"到达目标点（MARS）",mars_pick:"抓取积木（MARS）"};
  const behaviors = c.behaviors.map(b => {
    const translated = {...b,...localizeBehavior(b,"zh"),title:names[b.id] || b.title};
    if (b.id.includes("imitate")) return {...translated,description:"跟踪关键帧参考动作的关节姿态和旋转时序，在物理环境中完成动作。",successMetric:"检查参考姿态与旋转时序的复现误差，并观察能否稳定完成动作与落地。"};
    if (b.id==="one_leg") return {...translated,successMetric:"评估每轮平稳保持单脚站立的时长，并观察实际动作。"};
    return translated;
  });
  return {behaviors, templates:c.templates.map(p=> {
    const b=behaviors.find(b=>b.id===p.local?.behavior);
    return b ? {...p,title:b.title,description:b.description,success:b.successMetric} : p;
  })};
}

/** Resolve plan handoffs without ever treating a missing clip as a filename. */
export async function loadReferenceClip(
  query: URLSearchParams,
  loaders: { plan: (path: string) => Promise<Plan>; clip: (name: string) => Promise<Clip> },
): Promise<Clip | null> {
  const id = query.get("plan");
  if (id) {
    const revision = query.get("revision");
    if (revision !== null && (!/^[1-9]\d*$/.test(revision) || !Number.isSafeInteger(Number(revision)))) {
      throw new Error("方案版本必须是正整数");
    }
    const path = `/plans/${encodeURIComponent(id)}${revision === null ? "" : `/versions/${revision}`}`;
    const plan = await loaders.plan(path);
    if (!plan.reference) throw new Error("方案没有参考动作");
    return plan.reference;
  }
  const name = query.get("clip");
  return name ? loaders.clip(name) : null;
}
