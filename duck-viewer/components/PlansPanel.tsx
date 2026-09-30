"use client";
import Link from "next/link";
import {useEffect, useRef, useState} from "react";
import {listClips, loadClip, putClip, type Clip, type StoredClip} from "@/lib/anim";
import {clipPlan, localizedCatalog, downloadJSON, downloadText, newFromTemplate, planFile, planRequest, planNavigationQuery, pythonPlan, type Catalog, type Plan} from "@/lib/plans";
import s from "./PlansPanel.module.css";

type Library = {plans:Plan[]; path:string; clipPath:string; errors:{path:string;message:string}[]};
type Ai = {configured:boolean; base_url:string; model:string};
const cloudDefault = {task:"Mjlab-Velocity-Flat-MicroDuck",iterations:1000,envs:64,gpu:"T4",flavor:"l4x1"};

export default function PlansPanel(){
  const [catalog,setCatalog]=useState<Catalog>({templates:[],behaviors:[]});
  const [library,setLibrary]=useState<Library>({plans:[],path:"",clipPath:"",errors:[]});
  const [clips,setClips]=useState<StoredClip[]>([]);
  const [draft,setDraft]=useState<Plan|null>(null);
  const [baseline,setBaseline]=useState("");
  const [tab,setTab]=useState("templates");
  const [compute,setCompute]=useState("local");
  const [notice,setNotice]=useState("");
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);
  const [online,setOnline]=useState(false);
  const [ai,setAi]=useState<Ai>({configured:false,base_url:"https://api.openai.com/v1",model:""});
  const [aiKey,setAiKey]=useState("");
  const [prompt,setPrompt]=useState("");
  const [aiRobot,setAiRobot]=useState("microduck");
  const [query,setQuery]=useState("");
  const [job,setJob]=useState<{compute:string;revision:number;id:string}|null>(null);
  const fileRef=useRef<HTMLInputElement>(null);
  const clipRef=useRef<HTMLInputElement>(null);
  const initial=useRef(false);
  const dirty=!!draft && JSON.stringify(planFile(draft))!==baseline;
  const recipe=catalog.behaviors.find(b=>b.id===draft?.local?.behavior);
  const robots=[...new Set(catalog.behaviors.map(b=>b.robot))];
  const link=(path:string)=>{const p=planNavigationQuery(typeof window!=="undefined"?window.location.search:"");return path+(p.size?`?${p}`:"");};
  async function refresh(){
    const [c,l,cl,a]=await Promise.all([planRequest<Catalog>("/plans/templates"),planRequest<Library>("/plans"),listClips(),planRequest<Ai>("/plans/ai/settings")]);
    setCatalog(localizedCatalog(c));setLibrary(l);setClips(cl);setAi(a);setOnline(true);return c;
  }
  function select(p:Plan){
    setDraft(structuredClone(p));setBaseline(JSON.stringify(planFile(p)));setCompute(p.local?"local":"colab");setError("");
  }
  async function action(fn:()=>Promise<void>){
    setBusy(true);setError("");
    try{await fn();}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
  }
  useEffect(()=>{let active=true; Promise.all([planRequest<Catalog>("/plans/templates"),planRequest<Library>("/plans"),listClips(),planRequest<Ai>("/plans/ai/settings")]).then(async([c,l,cl,a])=>{
    if(!active)return;setCatalog(localizedCatalog(c));setLibrary(l);setClips(cl);setAi(a);setOnline(true);
    if(!initial.current){initial.current=true;const name=new URLSearchParams(window.location.search).get("clip");
      if(name){const clip=await loadClip(name);if(active){select(clipPlan(clip,localizedCatalog(c).behaviors));setNotice("参考动作已绑定。请检查本地配方和训练预算，再保存方案。");}}
    }
  }).catch(e=>{if(active)setError(`无法加载训练方案：${e.message}`);});return()=>{active=false;};},[]);
  useEffect(()=>{const warn=(e:BeforeUnloadEvent)=>{if(dirty){e.preventDefault();}};window.addEventListener("beforeunload",warn);return()=>window.removeEventListener("beforeunload",warn);},[dirty]);
  function switchDraft(p:Plan){if(dirty&&!window.confirm("当前方案有未保存的修改，是否放弃并打开另一方案？"))return;select(p);}
  function patch(fields:Partial<Plan>){setDraft(p=>p?{...p,...fields}:p);}
  async function save(copy=false){if(!draft)return;
    const payload=copy?newFromTemplate(draft):planFile(draft);
    const existing=payload.id&&!payload.id.startsWith("template-");
    const p=await planRequest<Plan>(existing?`/plans/${payload.id}`:"/plans",existing?"PUT":"POST",payload);
    select(p);await refresh();setTab("mine");setNotice(`已保存“${p.title}” · v${p.revision}。方案保存在本地，执行位置另行选择。`);
  }
  async function importFile(file:File,kind:"plan"|"clip"){
    if(file.size>2_000_000)throw new Error("文件超过 2 MB");
    const content=await file.text();
    if(kind==="plan"){
      const p=await planRequest<Plan>("/plans/import","POST",{filename:file.name,content});switchDraft(p);await refresh();setTab("mine");setNotice(`导入成功：${p.title}。可查看路径并选择执行位置。`);
    }else{
      const value=JSON.parse(content) as Clip;
      if(!value.name)value.name=file.name.replace(/\.json$/i,"");
      // Backend performs joint-count, time ordering and robot checks.
      const saved=await putClip(value);switchDraft(clipPlan(saved,catalog.behaviors));await refresh();setNotice(`动作已保存至 ${library.clipPath}。模仿方案仍需保存。`);
    }
  }
  const localTerms=recipe?[...recipe.terms,...recipe.availableTerms.filter(t=>draft?.local?.weights[t.key]!==undefined)]:[];
  return <main className={s.page}>
    <header className={s.header}><Link href={link("/")}>← 机器人实验室</Link><div><h1>训练方案</h1><small>定义目标 · 绑定动作 · 配置执行环境</small></div><div className={s.spacer}/><Link href={link("/train")}>训练记录</Link><Link href={link("/cloud")}>云端算力</Link><button disabled={busy} onClick={()=>action(async()=>{await refresh();setNotice("方案库已刷新");})}>刷新</button></header>
    <div className={s.flow}><span>01 选择或创建方案</span><span>02 检查本地 / 云端配置</span><span>03 保存版本</span><span>04 启动并观察训练</span></div>
    {error&&<div role="alert" className={s.error}>{error}</div>}{notice&&<div role="status" className={s.notice}>{notice}</div>}
    {job&&<div className={s.notice}>已启动方案 v{job.revision} · {job.compute==="local"?"本机 CPU":job.compute==="colab"?"Google Colab":"Hugging Face"} · {job.id}。 <Link href={link(job.compute==="local"?"/":"/cloud")}>查看进度、停止或管理训练 →</Link></div>}
    <div className={s.layout}>
      <aside className={s.sidebar}><h2>方案库</h2><p>训练方案决定如何学习；模型是训练结果；动作片段是模仿训练的参考。</p>
        <div className={s.actions}><button disabled={!online||busy} onClick={()=>fileRef.current?.click()}>↑ 导入训练方案</button><button disabled={!online||busy} onClick={()=>clipRef.current?.click()}>↑ 导入动作 JSON</button></div>
        <p>方案支持 .json / 含 TRAINING_PLAN 的 .py。任意 Python 脚本需先适配；.onnx 属于模型。</p>
        <input aria-label="上传训练方案文件" ref={fileRef} type="file" accept=".json,.py" hidden onChange={e=>{const f=e.target.files?.[0];e.target.value="";if(f)action(()=>importFile(f,"plan"));}}/>
        <input aria-label="上传动作片段文件" ref={clipRef} type="file" accept=".json" hidden onChange={e=>{const f=e.target.files?.[0];e.target.value="";if(f)action(()=>importFile(f,"clip"));}}/>
        <div className={s.mode}>{[["templates","已有模板"],["mine","我的方案"],["clips","关键帧动作"],["ai","AI 创建"]].map(([id,title])=><button key={id} aria-pressed={tab===id} onClick={()=>setTab(id)}>{title}</button>)}</div>
        <label>搜索<input value={query} onChange={e=>setQuery(e.target.value)} placeholder="方案名称 / 机器人"/></label>
        {tab!=="ai"&&<div className={s.list} style={{marginTop:14}}>
          {(tab==="clips"?clips.filter(c=>`${c.name} ${c.robot||"microduck"}`.toLowerCase().includes(query.toLowerCase())):(tab==="mine"?library.plans:catalog.templates).filter(p=>`${p.title} ${p.robot}`.toLowerCase().includes(query.toLowerCase()))).map(item=>"keys" in item?<button disabled={busy} key={item.name} className={s.card} onClick={()=>action(async()=>switchDraft(clipPlan(await loadClip(item.name),catalog.behaviors)))}><strong>🎬 {item.name}</strong><small>{item.keys.length} 个关键帧 · {item.duration} 秒 · {item.robot||"microduck"}</small><small>创建本地模仿训练方案 →</small></button>:<button disabled={busy} key={item.id} className={`${s.card} ${draft?.id===item.id?s.selected:""}`} onClick={()=>switchDraft(tab==="templates"?newFromTemplate(item):item)}><strong>{item.title}</strong><small>{item.robot} · {item.local?"本地 SB3":""}{item.local&&item.cloud?" / ":""}{item.cloud?"云端 mjlab":""}{item.revision?` · v${item.revision}`:""}</small></button>)}
          {tab==="mine"&&!library.plans.length&&<p className={s.empty}>还没有方案。选择模板、导入文件或从关键帧创建。</p>}
          {tab==="clips"&&!clips.length&&<p className={s.empty}>没有动作片段。到实验室的动画窗口制作关键帧，或导入兼容动作 JSON。</p>}
        </div>}
        {tab==="ai"&&<div className={s.section}><h3>AI 辅助创建</h3><p>根据现有训练能力生成可编辑草稿。新的环境或奖励函数仍需开发接入；AI 不会自动启动训练。</p><label>机器人<select value={aiRobot} onChange={e=>setAiRobot(e.target.value)}>{robots.map(r=><option key={r}>{r}</option>)}</select></label><label style={{marginTop:10}}>描述目标<textarea value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="例如：让 MicroDuck 单脚保持平衡，减少晃动"/></label><button className={s.primary} disabled={!ai.configured||busy||!prompt.trim()} onClick={()=>action(async()=>{const result=await planRequest<{draft:Plan;notice:string}>("/plans/ai/draft","POST",{prompt,robot:aiRobot});switchDraft(result.draft);setNotice(result.notice);})}>生成方案草稿</button>
          <details open={!ai.configured}><summary>{ai.configured?`模型服务 · ${ai.model}`:"配置 AI 模型服务"}</summary><p>支持 OpenAI 兼容服务及本机服务。远程服务会收到目标和可用模板，按服务商规则计费。</p><label>API 基础地址<input value={ai.base_url} onChange={e=>setAi({...ai,base_url:e.target.value})}/></label><label>模型名称<input value={ai.model} onChange={e=>setAi({...ai,model:e.target.value})}/></label><label>API 密钥<input type="password" autoComplete="off" value={aiKey} onChange={e=>setAiKey(e.target.value)} placeholder="已有密钥可留空；本机服务可不填"/></label><button disabled={busy||!ai.model} onClick={()=>action(async()=>{const result=await planRequest<Ai>("/plans/ai/settings","PUT",{base_url:ai.base_url,model:ai.model,api_key:aiKey});setAi(result);setAiKey("");setNotice("AI 服务配置已保存在本地。");})}>保存服务配置</button></details>
        </div>}
        <details className={s.section}><summary>文件保存位置</summary><p>方案与动作保存到当前连接的训练服务。云端训练使用独立任务配置。</p><code className={s.path}>{library.path||"连接后显示实际方案目录"}</code><code className={s.path}>{library.clipPath||"连接后显示实际动作目录"}</code><button disabled={!library.path} onClick={()=>action(async()=>{await navigator.clipboard.writeText(library.path);setNotice("方案目录已复制");})}>复制方案目录</button></details>
        {!!library.errors.length&&<div className={s.error}>有 {library.errors.length} 个方案文件无法读取，请检查文件格式。</div>}
      </aside>
      <section className={s.editor} aria-label="训练方案编辑器">{!draft?<><h2>先定义如何学习</h2><p className={s.muted}>从左侧选择模板、导入文件，或使用关键帧和 AI 创建方案。</p><div className={s.section}><h3>通过关键帧创建</h3><p className={s.muted}>在机器人实验室打开“动画”，调整姿态、添加关键帧并保存，然后选择“创建训练方案”。参考动画与物理训练结果需要分别观察。</p><Link className={s.button} href={link("/")}>打开机器人实验室 →</Link></div></>:<>
        <h2><span className={s.number}>01</span>目标与成功标准 <span className={s.badge}>{draft.revision?`v${draft.revision}`:"草稿"}{dirty?" · 未保存修改":""}</span></h2>
        <div className={s.grid}><label>方案名称<input maxLength={120} value={draft.title} onChange={e=>patch({title:e.target.value})}/></label><label>机器人<input value={draft.robot} readOnly/></label><label className={s.wide}>训练目标<textarea value={draft.description} onChange={e=>patch({description:e.target.value})}/></label><label className={s.wide}>成功标准<textarea value={draft.success} onChange={e=>patch({success:e.target.value})}/></label></div>
        {draft.reference&&<div className={s.section}><h3>参考动作</h3><div className={s.clip}><strong>🎬 {draft.reference.name}</strong><span className={s.muted}>{draft.reference.keys.length} 个关键帧 · {draft.reference.duration} 秒 · {draft.reference.loop?"循环":"单次"}</span><Link className={s.button} href={draft.id?`${link("/")}${link("/").includes("?")?"&":"?"}plan=${draft.id}&revision=${draft.revision}`:`${link("/")}${link("/").includes("?")?"&":"?"}clip=${encodeURIComponent(draft.reference.name)}`}>预览参考动作</Link><button onClick={()=>downloadJSON(draft.reference,`${draft.reference!.name}.json`)}>导出动作</button></div><p className={s.muted}>动作内容随方案版本保存。动画可以播放不代表机器人已在物理环境中学会。</p></div>}
        <section className={s.section}><h3><span className={s.number}>02</span>本地实现 · MuJoCo / SB3 / CPU</h3><label className={s.check}><input type="checkbox" checked={!!draft.local} onChange={e=>{if(!e.target.checked){patch({local:null});return;}const b=catalog.behaviors.find(b=>b.robot===draft.robot&&b.id===(draft.reference?(draft.robot==="microduck"?"imitate":`${draft.robot}_imitate`):"stand"))||catalog.behaviors.find(b=>b.robot===draft.robot&&!b.id.includes("imitate"));if(b)patch({local:{behavior:b.id,steps:b.defaultSteps,weights:Object.fromEntries(b.terms.map(t=>[t.key,t.weight])),stageSteps:{},stageWeights:{}}});}}/>配置本地训练配方</label>
          {draft.local?<><div className={s.grid} style={{marginTop:14}}><label>训练配方<select value={draft.local.behavior} onChange={e=>{const b=catalog.behaviors.find(b=>b.id===e.target.value)!;patch({local:{behavior:b.id,steps:b.curriculum.reduce((n,s)=>n+s.steps,0)||b.defaultSteps,weights:Object.fromEntries(b.terms.map(t=>[t.key,t.weight])),stageSteps:{},stageWeights:{}}});}}>{catalog.behaviors.filter(b=>b.robot===draft.robot&&(draft.reference?b.id.includes("imitate"):!b.id.includes("imitate"))).map(b=><option key={b.id} value={b.id}>{b.title}</option>)}</select></label><label>总训练步数<input type="number" min={100000} max={40000000} value={draft.local.steps} onChange={e=>patch({local:{...draft.local!,steps:Number(e.target.value)}})}/></label></div>
            <details><summary>奖励权重（惩罚项使用正权重，符号由配方定义）</summary><div className={s.terms} style={{marginTop:12}}>{localTerms.map(t=><label key={t.key} className={s.term}><span>{t.isPenalty?"−":"＋"} {t.key}<br/><small className={s.muted}>{t.friendly}</small></span><input aria-label={`权重 ${t.key}`} type="number" min={0} max={1000} step={.1} value={draft.local!.weights[t.key]??t.weight} onChange={e=>patch({local:{...draft.local!,weights:{...draft.local!.weights,[t.key]:Number(e.target.value)}}})}/></label>)}</div><label style={{marginTop:12}}>添加奖励 / 惩罚<select value="" onChange={e=>{const t=recipe?.availableTerms.find(t=>t.key===e.target.value);if(t)patch({local:{...draft.local!,weights:{...draft.local!.weights,[t.key]:t.weight}}});}}><option value="">选择条件库中的项目</option>{recipe?.availableTerms.filter(t=>draft.local!.weights[t.key]===undefined).map(t=><option key={t.key} value={t.key}>{t.key}</option>)}</select></label></details>
            {!!recipe?.curriculum.length&&<details><summary>分阶段课程 · 默认按总步数比例分配</summary>{recipe.curriculum.map((stage,i)=><label key={stage.label} style={{marginTop:10}}>{i+1}. {stage.label}<small>{stage.detail}</small><input aria-label={`阶段 ${i+1} 步数`} type="number" min={1} max={40000000} placeholder={`默认比例 · 原始 ${stage.steps}`} value={draft.local!.stageSteps[String(i+1)]??""} onChange={e=>{const values={...draft.local!.stageSteps};if(e.target.value)values[String(i+1)]=Number(e.target.value);else delete values[String(i+1)];patch({local:{...draft.local!,stageSteps:values}});}}/></label>)}</details>}
          </>:<p className={s.muted}>尚未配置本地实现，此方案不能在本机执行。</p>}
        </section>
        <section className={s.section}><h3>云端实现 · 官方 mjlab / MuJoCo Warp / GPU</h3><label className={s.check}><input type="checkbox" disabled={draft.robot!=="microduck"||!!draft.reference} checked={!!draft.cloud} onChange={e=>patch({cloud:e.target.checked?{...cloudDefault}:null})}/>单独配置官方云端任务</label><p className={s.muted}>{draft.reference?"关键帧模仿尚未接入官方云端框架，当前只能使用本地实现。":"云端任务独立于本地配方。此处选择官方任务，不会自动采用本地奖励权重或课程。"}</p>
          {draft.cloud&&<div className={s.grid} style={{marginTop:12}}><label className={s.wide}>官方训练任务<select value={draft.cloud.task} onChange={e=>patch({cloud:{...draft.cloud!,task:e.target.value}})}><option value="Mjlab-Velocity-Flat-MicroDuck">官方行走 · Velocity</option><option value="Mjlab-VelStand-Flat-MicroDuck">官方行走与跌倒恢复 · VelStand</option></select></label>{([["iterations","迭代次数",100000],["envs","并行环境数",4096]] as const).map(([key,label,max])=><label key={key}>{label}<input type="number" min={1} max={max} value={draft.cloud![key]} onChange={e=>patch({cloud:{...draft.cloud!,[key]:Number(e.target.value)}})}/></label>)}<label>Colab GPU<select value={draft.cloud.gpu} onChange={e=>patch({cloud:{...draft.cloud!,gpu:e.target.value}})}>{["T4","L4","A100"].map(g=><option key={g}>{g}</option>)}</select></label><label>Hugging Face 硬件规格<input value={draft.cloud.flavor} onChange={e=>patch({cloud:{...draft.cloud!,flavor:e.target.value}})}/><Link href={link("/cloud")}>查看可用硬件、连接账号 →</Link></label></div>}
        </section>
        <section className={s.section}><h3><span className={s.number}>03</span>保存与版本</h3><p className={s.muted}>每次保存创建独立版本，训练使用选定版本的快照。文件保存在本地训练服务。</p><code className={s.path}>{draft.path||`${library.path||"本地方案目录"}/〈方案 ID〉/v1.json（保存后显示实际路径）`}</code><div className={s.actions}><button disabled={busy||!online} onClick={()=>action(()=>save())}>{draft.id?"保存新版本":"保存方案"}</button><button disabled={busy||!online} onClick={()=>action(()=>save(true))}>另存为新方案</button><button onClick={()=>downloadJSON(planFile(draft),"training-plan.json")}>导出方案 JSON</button><button onClick={()=>downloadText(pythonPlan(draft),"training-plan.py")}>导出 Python 模板</button><button disabled={!draft.path} onClick={()=>action(async()=>{await navigator.clipboard.writeText(draft.path!);setNotice("文件路径已复制");})}>复制文件路径</button></div>
          {!!draft.revisions?.length&&<label>查看历史版本<select value={draft.revision} onChange={e=>action(async()=>switchDraft(await planRequest<Plan>(`/plans/${draft.id}/versions/${e.target.value}`)))}>{draft.revisions.map(v=><option key={v} value={v}>v{v}</option>)}</select></label>}
        </section>
        <section className={s.section}><h3><span className={s.number}>04</span>选择执行位置</h3><div className={s.mode}>{[["local","本机 CPU",!!draft.local],["colab","Google Colab",!!draft.cloud],["hf","Hugging Face",!!draft.cloud]].map(([id,label,enabled])=><button key={String(id)} disabled={!enabled||busy} aria-pressed={compute===id} onClick={()=>setCompute(String(id))}>{String(label)}{!enabled?" · 未适配":""}</button>)}</div><p className={s.muted}>{compute==="local"?`使用本地配方 ${draft.local?.behavior||"（未配置）"}。训练将在实验室显示。`:`使用官方任务 ${draft.cloud?.task||"（未配置）"}。云端费用与资源由服务商决定，请先连接账号。`}</p></section>
        <div className={s.footer}><p>{dirty||!draft.id?"保存方案后才能开始训练。":`将执行 v${draft.revision} 的${compute==="local"?"本地":"云端"}配置。`}</p><button className={s.primary} disabled={busy||dirty||!draft.id||!online||(compute==="local"?!draft.local:!draft.cloud)} onClick={()=>action(async()=>{const check=await planRequest<{ready:boolean;problems:string[]}>(`/plans/${draft.id}/check`,"POST",{revision:draft.revision,compute});if(!check.ready)throw new Error(check.problems.join("；"));const r=await planRequest<{revision:number;job:Record<string,string>}>(`/plans/${draft.id}/launch`,"POST",{revision:draft.revision,compute});setJob({compute,revision:r.revision,id:r.job.runName||r.job.run||r.job.id||draft.id!});setNotice("训练已提交，使用已保存的方案版本。");})}>{busy?"处理中…":"▶ 开始训练"}</button></div>
      </>}</section>
    </div>
  </main>;
}
