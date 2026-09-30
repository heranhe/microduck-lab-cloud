"""Versioned, declarative training plans. Imported Python is parsed, never executed.

A local recipe and an official mjlab task are independent implementations.
Plans embed their reference clip so export/import and later edits are reproducible.
"""
from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import uuid
from pathlib import Path
from threading import RLock
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import URLError

from fastapi import HTTPException, Request
from pydantic import BaseModel

MAX_BYTES = 2_000_000
TASKS = {
    "Mjlab-Velocity-Flat-MicroDuck": "官方行走",
    "Mjlab-VelStand-Flat-MicroDuck": "官方行走与跌倒恢复",
}


def root() -> Path:
    return Path(os.environ.get("MICRODUCK_PLANS_DIR") or
                Path(__file__).resolve().parents[2] / "training-plans")


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError("方案 ID 仅支持英文字母、数字、下划线和短横线")
    return value


def integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} 必须是 {low}–{high} 之间的整数")
    return value


def text_field(value, label, limit=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"请填写{label}（最多 {limit} 字符）")
    return value.strip()


def clean_plan(data, behaviors, clean_clip):
    if len(json.dumps(data).encode()) > MAX_BYTES:
        raise ValueError("方案内容超过 2 MB")
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("需要 version: 1 的训练方案；动作 JSON 请使用动作导入入口")
    allowed = {"version", "id", "revision", "title", "description", "success", "robot", "source", "local", "cloud", "reference"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError("无法识别的方案字段：" + ", ".join(sorted(unknown)))
    p = {"version": 1, "title": text_field(data.get("title"), "方案名称", 120),
         "description": text_field(data.get("description"), "训练目标"),
         "success": text_field(data.get("success"), "成功标准"),
         "robot": text_field(data.get("robot"), "机器人", 80),
         "source": data.get("source", "manual"), "local": None, "cloud": None}
    if p["source"] not in ("manual", "template", "import", "animation", "ai", "run"):
        raise ValueError("未知方案来源")
    reference = data.get("reference")
    if reference is not None:
        if not isinstance(reference, dict):
            raise ValueError("参考动作必须是 JSON 对象")
        reference = clean_clip(reference.get("name", "reference"), reference)
        if reference.get("robot", "microduck") != p["robot"]:
            raise ValueError("参考动作与方案的机器人不一致")
        p["reference"] = reference
    local = data.get("local")
    if local is not None:
        if not isinstance(local, dict) or set(local) - {"behavior", "steps", "weights", "stageSteps", "stageWeights"}:
            raise ValueError("本地配置支持 behavior、steps、weights、stageSteps、stageWeights")
        b = behaviors.BEHAVIORS.get(local.get("behavior"))
        if b is None or b.robot != p["robot"]:
            raise ValueError("本地配方不存在，或机器人不匹配")
        if b.id in ("imitate", "g1_imitate") and reference is None:
            raise ValueError("模仿训练必须绑定参考动作，不能回退到默认片段")
        if reference is not None and b.id not in ("imitate", "g1_imitate"):
            raise ValueError("参考动作只适用于模仿训练配方")
        keys = {t.key for t in b.terms} | set(behaviors.CATALOG)
        def weights(values):
            if not isinstance(values, dict):
                raise ValueError("奖励权重必须是对象")
            for k, v in values.items():
                if k not in keys or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000:
                    raise ValueError(f"无效奖励权重：{k}；数值必须在 0–1000 之间")
            return values
        # Pin every base term, not sticky weights from another run.
        w = {t.key: t.weight for t in b.terms}
        w.update(weights(local.get("weights", {})))
        l = {"behavior": b.id, "steps": integer(local.get("steps"), "训练步数", 100_000, 40_000_000),
             "weights": w, "stageSteps": {}, "stageWeights": {}}
        for field in ("stageSteps", "stageWeights"):
            values = local.get(field, {})
            if not isinstance(values, dict):
                raise ValueError(f"{field} 必须是对象")
            for k, v in values.items():
                if k not in {str(i + 1) for i in range(len(b.curriculum))}:
                    raise ValueError(f"未知课程阶段：{k}")
                l[field][k] = weights(v) if field == "stageWeights" else integer(v, "阶段步数", 1, 40_000_000)
        p["local"] = l
    cloud = data.get("cloud")
    if cloud is not None:
        if not isinstance(cloud, dict) or set(cloud) - {"task", "iterations", "envs", "gpu", "flavor"}:
            raise ValueError("云端配置支持 task、iterations、envs、gpu、flavor；不自动使用本地奖励或参考动作")
        if p["robot"] != "microduck" or cloud.get("task") not in TASKS:
            raise ValueError("云端仅支持已接入的官方 MicroDuck 任务")
        if reference is not None:
            raise ValueError("当前官方云端任务不支持关键帧模仿；请移除云端配置")
        p["cloud"] = {"task": cloud["task"],
                      "iterations": integer(cloud.get("iterations"), "云端迭代次数", 1, 100000),
                      "envs": integer(cloud.get("envs"), "并行环境数", 1, 4096),
                      "gpu": cloud.get("gpu", "T4"), "flavor": cloud.get("flavor", "l4x1")}
        for field in ("gpu", "flavor"):
            text_field(p["cloud"][field], field, 80)
    if p["local"] is None and p["cloud"] is None:
        raise ValueError("至少配置一种训练实现")
    return p


def parse_file(filename: str, content: str):
    if len(content.encode()) > MAX_BYTES:
        raise ValueError("文件超过 2 MB")
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return json.loads(content)
    if suffix == ".py":
        tree = ast.parse(content)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAINING_PLAN" for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    break
        raise ValueError("Python 文件需包含 TRAINING_PLAN = {...} 字面量。任意训练脚本需先适配配方，不能直接执行；可下载 Python 模板。")
    raise ValueError("支持训练方案 .json 或含 TRAINING_PLAN 的 .py；模型 .onnx 和检查点不是训练方案")


class ImportReq(BaseModel):
    filename: str
    content: str


class LaunchReq(BaseModel):
    revision: int
    compute: str


class AiReq(BaseModel):
    prompt: str
    robot: str = "microduck"


class AiSettings(BaseModel):
    base_url: str
    model: str
    api_key: str = ""


def mount_plans(app, *, behaviors, clean_clip, atomic_write, launch_local, launch_colab, launch_hf,
                teach_req, colab_req, hf_req, clip_path, origin_allowed):
    lock = RLock()
    def guard(request):
        if not origin_allowed(request.headers.get("origin")):
            raise HTTPException(403, "Origin not allowed")
    def validated(data):
        try:
            return clean_plan(data, behaviors, clean_clip)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
    def read(pid, revision=None):
        try:
            identifier(pid)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        path = root() / pid / (f"v{revision}.json" if revision else "plan.json")
        if not path.is_file():
            raise HTTPException(404, "未找到方案版本")
        return json.loads(path.read_text())
    def card(p):
        path = root() / p["id"] / f"v{p['revision']}.json"
        return {**p, "path": str(path.resolve()), "storage": "local",
                "revisions": sorted(int(f.stem[1:]) for f in path.parent.glob("v*.json"))}
    def save(data, pid=None, expected=None):
        p = validated(data)
        with lock:
            pid = identifier(pid or uuid.uuid4().hex[:12])
            rev = 1
            if (root() / pid / "plan.json").exists():
                old = read(pid)
                if expected != old["revision"]:
                    raise HTTPException(409, "方案已被修改，请重新加载后保存")
                rev = old["revision"] + 1
            p.update(id=pid, revision=rev)
            atomic_write(root() / pid / f"v{rev}.json", p)
            atomic_write(root() / pid / "plan.json", p)
        return card(p)

    @app.get("/plans")
    def list_plans():
        plans, errors = [], []
        for f in root().glob("*/plan.json"):
            try:
                plans.append(card(json.loads(f.read_text())))
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"path": str(f), "message": str(exc)})
        return {"plans": plans, "errors": errors, "path": str(root().resolve()),
                "clipPath": str(clip_path("reference").parent.resolve())}

    @app.get("/plans/templates")
    def templates():
        out = []
        for b in behaviors.BEHAVIORS.values():
            if b.id in ("imitate", "g1_imitate"):
                continue  # imitation needs an explicit clip
            c = behaviors.behavior_card(b)
            out.append({"version": 1, "id": "template-" + b.id, "title": b.title,
                        "robot": b.robot, "description": b.description,
                        "success": b.success_metric or "评估并观察实际动作是否完成目标",
                        "source": "template", "local": {"behavior": b.id,
                            "steps": sum(s.steps for s in b.curriculum) or b.default_steps,
                            "weights": {t.key: t.weight for t in b.terms}, "stageSteps": {}, "stageWeights": {}},
                        "cloud": None, "terms": c["terms"], "availableTerms": c["availableTerms"],
                        "curriculum": c["curriculum"]})
        for task, title in TASKS.items():
            out.append({"version": 1, "id": "template-" + task, "title": title,
                        "robot": "microduck", "description": title + "，使用官方 mjlab GPU 训练框架",
                        "success": "评估速度跟踪、稳定性与实际动作表现", "source": "template", "local": None,
                        "cloud": {"task": task, "iterations": 1000, "envs": 64, "gpu": "T4", "flavor": "l4x1"}})
        return {"templates": out, "behaviors": [{**behaviors.behavior_card(b), "robot": b.robot} for b in behaviors.BEHAVIORS.values()]}

    @app.post("/plans")
    def create_plan(data: dict, request: Request):
        guard(request)
        return save(data)

    @app.put("/plans/{pid}")
    def update_plan(pid: str, data: dict, request: Request):
        guard(request)
        return save(data, pid, data.get("revision"))

    @app.get("/plans/{pid}/versions/{revision}")
    def version(pid: str, revision: int):
        return card(read(pid, revision))

    @app.post("/plans/import")
    def import_plan(req: ImportReq, request: Request):
        guard(request)
        try:
            data = parse_file(req.filename, req.content)
            if isinstance(data, dict):
                data["source"] = "import"
            return save(data)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/plans/{pid}/check")
    def check_plan(pid: str, req: LaunchReq, request: Request):
        guard(request)
        p = read(pid, req.revision)
        validated(p)
        problems = []
        if req.compute == "local":
            if not p["local"]:
                problems.append("未配置本地实现")
        elif req.compute in ("colab", "hf"):
            if not p["cloud"]:
                problems.append("未配置官方云端任务")
            else:
                from .colab_jobs import teacher_checkpoints
                try:
                    teacher_checkpoints(p["cloud"]["task"])
                except RuntimeError as exc:
                    problems.append(str(exc))
        else:
            problems.append("未知执行位置")
        return {"ready": not problems, "problems": problems,
                "framework": "SB3 CPU" if req.compute == "local" else "官方 mjlab GPU"}

    @app.post("/plans/{pid}/launch")
    async def launch_plan(pid: str, req: LaunchReq, request: Request):
        guard(request)
        p = read(pid, req.revision)
        validated(p)  # dependencies/clip contract may have changed since save
        if req.compute == "local" and p["local"]:
            l = p["local"]
            clip = None
            if p.get("reference"):
                clip = f"plan-{pid}-v{req.revision}"
                atomic_write(clip_path(clip), clean_clip(clip, p["reference"]))
            result = await launch_local(teach_req(text=l["behavior"], robot=p["robot"], clip=clip,
                steps=l["steps"], weights=l["weights"], stageSteps=l["stageSteps"], stageWeights=l["stageWeights"]))
            if not result.get("matched"):
                raise HTTPException(409, result.get("message", "训练未启动"))
        elif req.compute in ("colab", "hf") and p["cloud"]:
            c = p["cloud"]
            if req.compute == "colab":
                result = await asyncio.to_thread(launch_colab,
                    colab_req(task=c["task"], gpu=c["gpu"], iterations=c["iterations"], envs=c["envs"]), request)
            else:
                result = await asyncio.to_thread(launch_hf,
                    hf_req(task=c["task"], flavor=c["flavor"], iterations=c["iterations"], envs=c["envs"]), request)
        else:
            raise HTTPException(422, "该方案没有适配所选执行位置；请先配置对应实现")
        job = result.get("job", result)
        receipt = {"plan": p, "compute": req.compute, "job": job}
        atomic_write(root() / pid / f"launch-{uuid.uuid4().hex[:12]}.json", receipt)
        # Bind the exact immutable version to the artifact as well as the receipt.
        name = job.get("runName") if req.compute == "local" else job.get("id")
        if name and req.compute == "local":
            from . import viz_server
            atomic_write(viz_server.RUNS_DIR / name / "training-plan.json", p)
        return {"started": True, "planId": pid, "revision": req.revision, "compute": req.compute, "job": job}

    def ai_config():
        f = root() / "ai-settings.json"
        if f.exists():
            return json.loads(f.read_text())
        return {"base_url": os.environ.get("MICRODUCK_AI_BASE_URL", "https://api.openai.com/v1"),
                "model": os.environ.get("MICRODUCK_AI_MODEL", ""),
                "api_key": os.environ.get("MICRODUCK_AI_API_KEY", "")}

    @app.get("/plans/ai/settings")
    def get_ai_settings():
        c = ai_config()
        return {"configured": bool(c["model"]), "base_url": c["base_url"], "model": c["model"]}

    @app.put("/plans/ai/settings")
    def put_ai_settings(req: AiSettings, request: Request):
        guard(request)
        if not req.base_url.startswith(("https://", "http://localhost:", "http://127.0.0.1:")):
            raise HTTPException(422, "AI 地址需使用 HTTPS 或本机 HTTP 服务")
        try:
            text_field(req.model, "模型名称", 120)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        d = req.model_dump()
        if not d["api_key"]:
            old = ai_config()
            if old["base_url"] == d["base_url"]:
                d["api_key"] = old["api_key"]
        atomic_write(root() / "ai-settings.json", d, mode=0o600)
        return {"configured": True, "model": d["model"], "base_url": d["base_url"]}

    @app.post("/plans/ai/draft")
    def ai_draft(req: AiReq, request: Request):
        guard(request)
        c = ai_config()
        if not c["model"]:
            raise HTTPException(409, "请先配置 AI 模型服务")
        if not req.prompt.strip() or len(req.prompt) > 4000:
            raise HTTPException(422, "请填写目标，最多 4000 字符")
        candidates = [p for p in templates()["templates"] if p["robot"] == req.robot]
        system = ("Return only a JSON training plan with version,title,description,success,robot,source,local,cloud. "
                  "Use only an existing behavior/task from the supplied templates. Never invent reward keys, "
                  "Python code or cloud adaptation. For an unsupported skill explain in description what the "
                  "closest template does and what still needs authored keyframes or a developer extension. "
                  "Local and cloud are independent. Use Chinese user-facing text. Templates: " + json.dumps(candidates, ensure_ascii=False))
        payload = json.dumps({"model": c["model"], "messages": [{"role": "system", "content": system},
                    {"role": "user", "content": req.prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if c["api_key"]:
            headers["Authorization"] = "Bearer " + c["api_key"]
        try:
            with urlopen(UrlRequest(c["base_url"].rstrip("/") + "/chat/completions", payload, headers), timeout=45) as response:
                raw = json.load(response)["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            p = json.loads(raw)
            p["source"] = "ai"
            return {"draft": validated(p), "notice": "AI 草稿尚未保存或启动训练，请检查目标与配方是否相符。"}
        except HTTPException:
            raise
        except (URLError, ValueError, KeyError, IndexError, TypeError, TimeoutError) as exc:
            # Never return provider responses/headers that could expose the key.
            raise HTTPException(502, "AI 服务未生成可用方案，请检查服务连接或调整目标") from exc
