import {describe,it,expect} from "vitest";
import {clipPlan,newFromTemplate,planFile,planNavigationQuery,pythonPlan,type Plan,type Recipe} from "./plans";
const p:Plan={version:1,id:"abc",revision:2,title:"true null False",description:"目标",success:"标准",robot:"microduck",source:"manual",local:{behavior:"stand",steps:123,weights:{},stageSteps:{},stageWeights:{}},cloud:null,path:"/private/path",storage:"local",revisions:[1,2]};
describe("training plans",()=>{
  it("clears prior reference handoffs while preserving the selected backend",()=>{const query=planNavigationQuery("?lab=127.0.0.1%3A8799&plan=old&revision=2&clip=old-motion");query.set("plan","new");query.set("revision","3");expect(query.get("lab")).toBe("127.0.0.1:8799");expect(query.getAll("plan")).toEqual(["new"]);expect(query.getAll("revision")).toEqual(["3"]);expect(query.has("clip")).toBe(false);});
  it("exports portable configuration without storage metadata",()=>{expect(planFile(p)).not.toHaveProperty("path");expect(planFile(p).revision).toBe(2);});
  it("copies without identity and does not mutate existing weights",()=>{const q=newFromTemplate(p);q.local!.weights.test=2;expect(q.id).toBeUndefined();expect(p.local!.weights).toEqual({});});
  it("exports Python literals without altering strings",()=>{const value=pythonPlan({...p,reference:{version:1,name:"test",duration:1,loop:false,keys:[]}});expect(value).toContain('"title": "true null False"');expect(value).toContain('"loop": False');expect(value).toContain('"cloud": None');});
  it("requires a compatible imitation recipe",()=>{expect(()=>clipPlan({version:1,name:"test",duration:1,loop:false,keys:[]},[])).toThrow();});
  it("creates local-only imitation with an embedded reference",()=>{const b={id:"imitate",robot:"microduck",defaultSteps:100,successMetric:"匹配动作",terms:[{key:"pose_match",weight:4}]} as Recipe;const clip={version:1 as const,name:"test",duration:1,loop:false,keys:[]};const result=clipPlan(clip,[b]);expect(result.cloud).toBeNull();expect(result.reference).toEqual(clip);expect(result.local!.weights.pose_match).toBe(4);});
});
