import {describe,it,expect,vi} from "vitest";
import {loadReferenceClip,clipPlan,newFromTemplate,planFile,planNavigationQuery,pythonPlan,type Plan,type Recipe} from "./plans";
const p:Plan={version:1,id:"abc",revision:2,title:"true null False",description:"目标",success:"标准",robot:"microduck",source:"manual",local:{behavior:"stand",steps:123,weights:{},stageSteps:{},stageWeights:{}},cloud:null,path:"/private/path",storage:"local",revisions:[1,2]};
describe("training plans",()=>{
  it("clears prior reference handoffs while preserving the selected backend",()=>{const query=planNavigationQuery("?lab=127.0.0.1%3A8799&plan=old&revision=2&clip=old-motion");query.set("plan","new");query.set("revision","3");expect(query.get("lab")).toBe("127.0.0.1:8799");expect(query.getAll("plan")).toEqual(["new"]);expect(query.getAll("revision")).toEqual(["3"]);expect(query.has("clip")).toBe(false);});
  it("exports portable configuration without storage metadata",()=>{expect(planFile(p)).not.toHaveProperty("path");expect(planFile(p).revision).toBe(2);});
  it("copies without identity and does not mutate existing weights",()=>{const q=newFromTemplate(p);q.local!.weights.test=2;expect(q.id).toBeUndefined();expect(p.local!.weights).toEqual({});});
  it("exports Python literals without altering strings",()=>{const value=pythonPlan({...p,reference:{version:1,name:"test",duration:1,loop:false,keys:[]}});expect(value).toContain('"title": "true null False"');expect(value).toContain('"loop": False');expect(value).toContain('"cloud": None');});
  it("requires a compatible imitation recipe",()=>{expect(()=>clipPlan({version:1,name:"test",duration:1,loop:false,keys:[]},[])).toThrow();});
  it("creates local-only imitation with an embedded reference",()=>{const b={id:"imitate",robot:"microduck",defaultSteps:100,successMetric:"匹配动作",terms:[{key:"pose_match",weight:4}]} as Recipe;const clip={version:1 as const,name:"test",duration:1,loop:false,keys:[]};const result=clipPlan(clip,[b]);expect(result.cloud).toBeNull();expect(result.reference).toEqual(clip);expect(result.local!.weights.pose_match).toBe(4);});
});

const reference = {version:1 as const,name:"motion",duration:1,loop:false,keys:[]};
describe("reference handoff",()=>{
  it("loads the latest plan when no revision or clip is provided",async()=>{
    const plan=vi.fn(async()=>({...p,reference}));const clip=vi.fn();
    expect(await loadReferenceClip(new URLSearchParams("plan=abc"),{plan,clip})).toEqual(reference);
    expect(plan).toHaveBeenCalledWith("/plans/abc");expect(clip).not.toHaveBeenCalled();
  });
  it("loads the pinned revision and gives plan handoff precedence over clip",async()=>{
    const plan=vi.fn(async()=>({...p,reference}));const clip=vi.fn();
    await loadReferenceClip(new URLSearchParams("plan=abc&revision=2&clip=other"),{plan,clip});
    expect(plan).toHaveBeenCalledWith("/plans/abc/versions/2");expect(clip).not.toHaveBeenCalled();
  });
  it.each(["", "0", "-1", "1.5", "NaN", "9007199254740992"])("rejects invalid revision %s before any request",async revision=>{
    const plan=vi.fn();const clip=vi.fn();
    await expect(loadReferenceClip(new URLSearchParams({plan:"abc",revision}),{plan,clip})).rejects.toThrow("方案版本必须是正整数");
    expect(plan).not.toHaveBeenCalled();expect(clip).not.toHaveBeenCalled();
  });
  it("reports missing reference without falling back to a null clip",async()=>{
    const clip=vi.fn();
    await expect(loadReferenceClip(new URLSearchParams("plan=abc"),{plan:async()=>p,clip})).rejects.toThrow("方案没有参考动作");
    expect(clip).not.toHaveBeenCalled();
  });
  it("still supports direct clip handoff",async()=>{
    const plan=vi.fn();const clip=vi.fn(async()=>reference);
    expect(await loadReferenceClip(new URLSearchParams("clip=motion"),{plan,clip})).toEqual(reference);
    expect(clip).toHaveBeenCalledWith("motion");expect(plan).not.toHaveBeenCalled();
  });
  it("makes no requests for an absent handoff",async()=>{
    const plan=vi.fn();const clip=vi.fn();
    expect(await loadReferenceClip(new URLSearchParams(),{plan,clip})).toBeNull();
    expect(plan).not.toHaveBeenCalled();expect(clip).not.toHaveBeenCalled();
  });
});
