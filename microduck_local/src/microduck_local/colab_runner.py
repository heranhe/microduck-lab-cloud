"""Colab 云端 GPU 训练驱动器。

对外暴露 ColabTrainingJob，接口与 viz_server.TrainingJob 对齐：
  - status, progress, run_name, restarting, owns_preview_ducks  (属性)
  - poll()        → (progress_changed: bool, new_snapshot: bool)
  - stop()
  - payload()     → dict（与 TrainingJob.payload() 格式一致，多 "backend": "colab"）
  - display_title()
  - stage_payload() → None（Colab 暂不支持课程分阶段，单任务单运行）

工作流：
  1. __init__: 生成本地 run 目录 + 训练 Python 脚本
  2.           colab new -s <session> --gpu T4    分配云端 GPU
  3.           colab exec -s <session> -f <script> 启动远程训练
  4. _poll_loop (asyncio 后台任务):
               每 30 s 执行一次:
                 a. colab exec 读取远端 progress.jsonl 末几行 → 更新 self.progress
                 b. colab download live.onnx        → 写入本地 run 目录
               poll() 调用时检查文件变化, 返回 (changed, snap)
  5. stop():   colab stop -s <session>
  6. 训练结束后 (status→done): colab download 全量产物 + 同步 Google Drive
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # 仅类型检查时用


def _resolve_colab_bin() -> str:
    """自动解析 colab 可执行文件路径（优先匹配当前虚拟环境，做到完全开箱即用）。"""
    # 1. 优先检查当前 Python 虚拟环境（如 uv/venv 的 .venv/bin/colab）
    venv_bin = Path(sys.executable).parent / "colab"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    # 2. 检查系统 PATH
    which_bin = shutil.which("colab")
    if which_bin:
        return which_bin
    # 3. 检查用户主目录 ~/.local/bin/colab
    user_local = Path.home() / ".local" / "bin" / "colab"
    if user_local.exists() and os.access(user_local, os.X_OK):
        return str(user_local)
    # 4. 常见系统级路径备选
    for p in ["/usr/local/bin/colab", "/opt/homebrew/bin/colab"]:
        if Path(p).exists():
            return p
    return "colab"


# colab CLI 路径
COLAB_BIN = _resolve_colab_bin()

# Google Drive 备份目标文件夹（Drive 根目录下的路径，自动创建）
DRIVE_BACKUP_FOLDER = "MicroDuck/training-backups"

# 轮询间隔（秒）：每 30 s 下载一次进度和快照
POLL_INTERVAL_S = 30

# 每次通过 colab exec 读取的最后 N 行 progress.jsonl
PROGRESS_TAIL_LINES = 5

# Colab 支持的计算加速卡及其估算费率表（单位：Compute Units / 小时）
COLAB_ACCELERATORS: list[dict] = [
    {
        "id": "T4",
        "name": "NVIDIA T4",
        "rate": 1.96,
        "desc": "经济入门 · 约 1.96 点/小时",
        "recommended": True,
    },
    {
        "id": "L4",
        "name": "NVIDIA L4",
        "rate": 4.5,
        "desc": "高性价比 · 约 4.5 点/小时",
        "recommended": False,
    },
    {
        "id": "A100",
        "name": "NVIDIA A100",
        "rate": 13.0,
        "desc": "旗舰性能 · 约 13 点/小时",
        "recommended": False,
    },
    {
        "id": "H100",
        "name": "NVIDIA H100",
        "rate": 25.0,
        "desc": "极致算力 · 约 25 点/小时",
        "recommended": False,
    },
    {
        "id": "CPU",
        "name": "CPU 运行时",
        "rate": 0.0,
        "desc": "免费备用 · 0 点/小时",
        "recommended": False,
    },
]


def get_colab_accelerators() -> list[dict]:
    """返回 Colab 支持的加速卡类型及费率字典。"""
    return list(COLAB_ACCELERATORS)


def colab_available() -> bool:
    """检查 colab CLI 是否可用（已安装且已授权）。"""
    if not COLAB_BIN or not Path(COLAB_BIN).exists():
        return False
    try:
        result = subprocess.run(
            [COLAB_BIN, "whoami"],
            capture_output=True, text=True, timeout=10
        )
        # whoami 返回 email 行时说明已授权
        return result.returncode == 0 and "@" in result.stdout
    except Exception:
        return False


def colab_whoami() -> dict:
    """返回 {email, ok}，cli 不可用时 ok=False。"""
    try:
        result = subprocess.run(
            [COLAB_BIN, "whoami"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # 解析输出：通常格式为 "email: xxx@gmail.com"
            lines = result.stdout.strip().splitlines()
            email = ""
            for line in lines:
                if "@" in line:
                    # 兼容 "email: x" 或直接一行 email
                    email = line.split(":", 1)[-1].strip()
                    break
            return {"ok": True, "email": email}
    except Exception:
        pass
    return {"ok": False, "email": ""}


# ---------------------------------------------------- 🔑 纯网页 OAuth 认证向导
# 客户端配置与 google-colab-cli 官方开源包一致，直接生成官方授权 URL 与换取凭证
COLAB_CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
COLAB_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
COLAB_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
COLAB_TOKEN_PATH = Path(os.path.expanduser("~/.config/colab-cli/token.json"))
COLAB_SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]


def get_colab_oauth_url() -> str:
    """生成用于在浏览器中打开的 Google 官方授权页面 URL。"""
    import urllib.parse
    params = {
        "client_id": COLAB_CLIENT_ID,
        "redirect_uri": COLAB_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(COLAB_SCOPES),
        "prompt": "consent",
        "access_type": "offline",
        "token_usage": "remote",
    }
    return "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)


def exchange_colab_code(code: str) -> dict:
    """用用户在网页回填的 authorization code 换取 token 并持久化到 ~/.config/colab-cli/token.json。"""
    import urllib.request
    import urllib.parse
    from datetime import datetime, timezone, timedelta

    cleaned_code = code.strip()
    if not cleaned_code:
        raise ValueError("授权码不能为空")

    data = urllib.parse.urlencode({
        "code": cleaned_code,
        "client_id": COLAB_CLIENT_ID,
        "client_secret": COLAB_CLIENT_SECRET,
        "redirect_uri": COLAB_REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Google 授权码兑换失败，请确认授权码是否正确或已过期: {e}")

    expires_in = token_data.get("expires_in", 3600)
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()

    creds_json = {
        "token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": COLAB_CLIENT_ID,
        "client_secret": COLAB_CLIENT_SECRET,
        "scopes": COLAB_SCOPES,
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": expiry,
    }

    COLAB_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLAB_TOKEN_PATH.write_text(json.dumps(creds_json, indent=2), encoding="utf-8")
    os.chmod(COLAB_TOKEN_PATH, 0o600)

    who = colab_whoami()
    return {"configured": who["ok"], "email": who.get("email", "")}


def delete_colab_token() -> bool:
    """断开 Google 账号连接，删除本地 token 文件。"""
    if COLAB_TOKEN_PATH.exists():
        COLAB_TOKEN_PATH.unlink()
        return True
    return False


def kill_all_colab_sessions() -> list[str]:
    """紧急关停所有活跃的 Colab 会话，防止扣费熔断保护。"""
    killed = []
    if not colab_available():
        return killed
    try:
        res = _run_colab(["sessions"], timeout=20.0)
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            for line in lines:
                parts = line.split()
                if parts and not parts[0].startswith("Session") and not parts[0].startswith("-"):
                    sess_name = parts[0]
                    stop_res = _run_colab(["stop", "-s", sess_name], timeout=30.0)
                    if stop_res.returncode == 0:
                        killed.append(sess_name)
    except Exception as e:
        print(f"[ColabJob] kill_all_colab_sessions 异常: {e}", flush=True)
    return killed


def _run_colab(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    """执行 colab CLI 命令，统一设置 PATH 和超时。"""
    env = {**os.environ, "PATH": str(Path(COLAB_BIN).parent) + ":" + os.environ.get("PATH", "")}
    return subprocess.run(
        [COLAB_BIN] + args,
        capture_output=True, text=True, timeout=timeout, env=env
    )


def _generate_training_script(
    task_id: str,
    run_name: str,
    num_envs: int,
    max_iterations: int,
    run_dir_remote: str,
    gpu_type: str = "T4",
) -> str:
    """生成在 Colab 云端执行的 Python 训练脚本（字符串形式）。

    脚本逻辑：
      1. 安装 microduck_rl 依赖（uv pip install）
      2. 克隆/使用 microduck_rl 训练栈
      3. 启动 `uv run train`，每 250 步保存 live.onnx 和 progress.jsonl
      4. 结束后导出 policy.onnx
    """
    # 使用 textwrap.dedent 保证缩进正确
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # 由 microduck-lab colab_runner.py 自动生成
        # 任务: {task_id}, 运行名: {run_name}

        import os, subprocess, sys, json, shutil, time
        from pathlib import Path

        # 禁用 W&B（避免需要 API Key 报错）
        os.environ["WANDB_MODE"] = "disabled"
        # 环境变量：告知训练脚本运行目录
        RUN_DIR = Path("{run_dir_remote}")
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        print("[colab-runner] 开始安装依赖...", flush=True)
        # 安装 uv（如未安装）
        if not shutil.which("uv"):
            subprocess.run(
                ["pip", "install", "uv", "--quiet"], check=True
            )

        # 克隆 microduck_rl（如未克隆）
        REPO_DIR = Path("/content/microduck_rl")
        if not REPO_DIR.exists():
            print("[colab-runner] 克隆 microduck_rl...", flush=True)
            subprocess.run([
                "git", "clone", "--depth=1",
                "https://github.com/Pollen-Robotics/microduck_rl.git",
                str(REPO_DIR)
            ], check=True)
        else:
            print("[colab-runner] 使用已有 microduck_rl 目录", flush=True)

        # 安装项目依赖
        print("[colab-runner] 安装 microduck_rl 依赖...", flush=True)
        subprocess.run(["uv", "sync", "--no-dev"], cwd=str(REPO_DIR), check=True)

        print(f"[colab-runner] 开始训练任务: {task_id}", flush=True)
        print(f"[colab-runner] 并行环境数: {num_envs}", flush=True)
        print(f"[colab-runner] 最大迭代数: {max_iterations}", flush=True)

        # 探测实际分配的显卡型号并持久化
        actual_gpu = "{gpu_type}"
        try:
            p_gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if p_gpu.returncode == 0 and p_gpu.stdout.strip():
                actual_gpu = p_gpu.stdout.strip().splitlines()[0]
        except Exception:
            pass
        hw_info = {{"requested_gpu": "{gpu_type}", "actual_gpu": actual_gpu}}
        (RUN_DIR / "hardware.json").write_text(json.dumps(hw_info), encoding="utf-8")
        print(f"[colab-runner] 🖥 实际分配显卡硬件: {{actual_gpu}}", flush=True)

        # 启动训练
        env = {{
            **os.environ,
            "WANDB_MODE": "disabled",
            "MICRODUCK_RUNS_DIR": str(RUN_DIR),
            "MICRODUCK_SNAP_STEPS": "250",   # 每 250 步保存 live.onnx
        }}

        cmd = [
            "uv", "run", "train",
            "--task", "{task_id}",
            "--run-name", "{run_name}",
            "--num-envs", "{num_envs}",
            "--max-iterations", "{max_iterations}",
        ]

        print(f"[colab-runner] 执行命令: {{' '.join(cmd)}}", flush=True)
        result = subprocess.run(cmd, cwd=str(REPO_DIR), env=env)

        if result.returncode == 0:
            print("[colab-runner] ✅ 训练完成！", flush=True)
            # 导出 ONNX
            export_cmd = ["uv", "run", "python", "scripts/export.py",
                          "--run", str(RUN_DIR / "{run_name}")]
            subprocess.run(export_cmd, cwd=str(REPO_DIR), env=env)
            print("[colab-runner] ✅ ONNX 已导出", flush=True)
        else:
            print(f"[colab-runner] ❌ 训练失败，返回码: {{result.returncode}}", flush=True)

        # 写入完成标记
        (RUN_DIR / "done.flag").write_text(
            json.dumps({{"status": "done" if result.returncode == 0 else "failed",
                         "timestamp": time.time()}})
        )
        print("[colab-runner] 完成标记已写入", flush=True)
    """)


class ColabTrainingJob:
    """Colab 云端 GPU 训练任务，镜像 TrainingJob 公共接口。

    viz_server 的 LabState.job 可持有 TrainingJob 或 ColabTrainingJob，
    WebSocket 帧、TeachPanel 和 3D 鸭子热更新无需区分两者。
    """

    # 与 TrainingJob 对齐的类级别默认值
    owns_preview_ducks: bool = False  # 构造时设为 True，adopt 路径不改

    def __init__(
        self,
        behavior_id: str,
        behavior_title: str,
        behavior_emoji: str,
        task_id: str,               # microduck_rl 任务 ID，如 "Mjlab-Velocity-Flat-MicroDuck"
        runs_dir: Path,             # 本地 RUNS_DIR，run 目录在此下创建
        num_envs: int = 64,
        max_iterations: int = 5000,
        gpu_type: str = "T4",       # 当前账号只支持 T4
    ):
        self.behavior_id = behavior_id
        self.behavior_title = behavior_title
        self.behavior_emoji = behavior_emoji
        self.task_id = task_id
        self.gpu_type = gpu_type
        self.actual_gpu = gpu_type
        self.num_envs = num_envs
        self.max_iterations = max_iterations

        # 运行身份
        self.run_name = f"colab-{behavior_id}-{uuid.uuid4().hex[:6]}"
        self.dir = runs_dir / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)

        # Colab 会话名（比 run_name 短，colab CLI 可能有限制）
        self._session_name = f"duck-{uuid.uuid4().hex[:8]}"

        # 与 TrainingJob 对齐的公共状态
        self.status = "training"          # training | done | stopped | failed
        self.progress: dict = {
            "steps": 0,
            "total": max_iterations * num_envs,  # 估算总步数
        }
        self.restarting = False
        self.owns_preview_ducks = True

        # 内部状态
        self._t0 = time.time()
        self._elapsed_final: float | None = None
        self._live_mtime = 0.0            # 本地 live.onnx 的最后修改时间
        self._progress_offset = 0         # 远端 progress.jsonl 已读字节偏移（估算行数）
        self._poll_task: asyncio.Task | None = None  # 后台轮询协程
        self._progress_changed = False     # poll() 消费标志
        self._new_snapshot = False         # poll() 消费标志

        # 记录 colab 会话是否成功创建
        self._session_started = False
        self._launch()

    # ------------------------------------------------------------------ 启动

    def _launch(self) -> None:
        """在 Colab 云端分配 GPU/CPU 并启动训练脚本（同步）。"""
        # 生成训练脚本并写到本地临时目录
        remote_run_dir = f"/content/runs/{self.run_name}"
        script_content = _generate_training_script(
            task_id=self.task_id,
            run_name=self.run_name,
            num_envs=self.num_envs,
            max_iterations=self.max_iterations,
            run_dir_remote=remote_run_dir,
            gpu_type=self.gpu_type,
        )
        self._script_path = self.dir / "colab_train.py"
        self._script_path.write_text(script_content, encoding="utf-8")
        self._remote_run_dir = remote_run_dir

        try:
            # 1. 创建 Colab 会话（分配 GPU 或 CPU）
            print(f"[ColabJob] 正在分配 Colab 计算资源 ({self.gpu_type})...", flush=True)
            if self.gpu_type.upper() == "CPU":
                result = _run_colab(
                    ["new", "-s", self._session_name],
                    timeout=120.0
                )
            else:
                result = _run_colab(
                    ["new", "-s", self._session_name, "--gpu", self.gpu_type],
                    timeout=120.0
                )
            if result.returncode != 0:
                err_msg = (result.stderr.strip() or result.stdout.strip())
                # 检查是否为配额不足/后端拒绝该型号
                if any(k in err_msg.lower() for k in ["rejected", "quota", "invalid", "unavailable", "400"]):
                    raise RuntimeError(
                        f"Google Colab 拒绝分配 {self.gpu_type}（当前账号可能暂无该显卡配额）。"
                        f"建议在界面中切换为 T4 或 CPU 运行时重试。"
                    )
                raise RuntimeError(f"colab new 失败: {err_msg}")
            self._session_started = True
            print(f"[ColabJob] 会话 {self._session_name} ({self.gpu_type}) 已创建", flush=True)

            # 2. 上传并执行训练脚本（后台运行，不等待完成）
            print("[ColabJob] 上传并启动训练脚本...", flush=True)
            exec_result = _run_colab(
                ["exec", "-s", self._session_name, "-f", str(self._script_path)],
                timeout=300.0   # 安装依赖最多 5 分钟
            )
            if exec_result.returncode != 0:
                raise RuntimeError(
                    f"colab exec 启动失败: "
                    f"{exec_result.stderr.strip() or exec_result.stdout.strip()}"
                )
            print("[ColabJob] 训练脚本已在云端启动", flush=True)

        except Exception as e:
            print(f"[ColabJob] ❌ 启动失败: {e}", flush=True)
            self.status = "failed"
            self._write_event(f"Colab 启动失败: {e}")
            return

        # 3. 启动异步后台轮询（需要在事件循环中执行）
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._poll_task = loop.create_task(self._poll_loop())
            else:
                # 同步上下文（测试）中跳过
                pass
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ 后台轮询

    async def _poll_loop(self) -> None:
        """每 30 秒从 Colab 云端拉取进度和 live.onnx 快照。"""
        while self.status == "training":
            await asyncio.sleep(POLL_INTERVAL_S)
            if self.status != "training":
                break
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._fetch_remote_state)
            except Exception as e:
                print(f"[ColabJob] 轮询出错（将重试）: {e}", flush=True)

    def _fetch_remote_state(self) -> None:
        """同步拉取远端进度和快照（在线程池中执行）。"""
        # ---- 拉取 progress.jsonl（读取末尾几行）
        progress_cmd = "\n".join([
            f"tail -n {PROGRESS_TAIL_LINES} {self._remote_run_dir}/progress.jsonl 2>/dev/null || echo ''"
        ])
        try:
            result = _run_colab(
                ["exec", "-s", self._session_name, "-c", progress_cmd],
                timeout=30.0
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("["):
                        continue  # 跳过 colab 自身的日志行
                    try:
                        rec = json.loads(line)
                        self.progress = {**self.progress, **rec}
                        self._progress_changed = True
                    except (json.JSONDecodeError, ValueError):
                        pass
        except Exception as e:
            print(f"[ColabJob] 读取进度失败: {e}", flush=True)

        # ---- 下载 live.onnx
        try:
            local_live = self.dir / "live.onnx"
            result = _run_colab(
                ["download", "-s", self._session_name,
                 f"{self._remote_run_dir}/live.onnx",
                 str(local_live)],
                timeout=60.0
            )
            if result.returncode == 0 and local_live.exists():
                self._new_snapshot = True
                print(f"[ColabJob] 新快照已下载 → {local_live}", flush=True)
        except Exception as e:
            print(f"[ColabJob] 下载 live.onnx 失败: {e}", flush=True)

        # ---- 拉取 hardware.json（探测实际分配显卡硬件）
        if not (self.dir / "hardware.json").exists():
            try:
                local_hw = self.dir / "hardware.json"
                result = _run_colab(
                    ["download", "-s", self._session_name,
                     f"{self._remote_run_dir}/hardware.json",
                     str(local_hw)],
                    timeout=20.0
                )
                if result.returncode == 0 and local_hw.exists():
                    try:
                        hw_data = json.loads(local_hw.read_text(encoding="utf-8"))
                        if hw_data.get("actual_gpu"):
                            self.actual_gpu = hw_data["actual_gpu"]
                            print(f"[ColabJob] 🖥 探测到云端实际分配硬件: {self.actual_gpu}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass

        # ---- 检查完成标记
        try:
            check_cmd = f"cat {self._remote_run_dir}/done.flag 2>/dev/null || echo ''"
            result = _run_colab(
                ["exec", "-s", self._session_name, "-c", check_cmd],
                timeout=15.0
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            flag = json.loads(line)
                            if "status" in flag:
                                final_status = flag["status"]  # "done" or "failed"
                                print(f"[ColabJob] 训练结束，状态: {final_status}", flush=True)
                                self.status = final_status
                                self._freeze_elapsed()
                                # 下载最终产物并同步 Drive
                                self._finalize(success=(final_status == "done"))
                        except (json.JSONDecodeError, ValueError):
                            pass
        except Exception as e:
            print(f"[ColabJob] 检查完成标记失败: {e}", flush=True)

    # ------------------------------------------------------------------ 完成后处理

    def _finalize(self, success: bool) -> None:
        """训练完成后：下载全量产物 + 同步 Google Drive。"""
        artifacts = [
            "policy.onnx", "model_final.pt", "velocity.onnx",
            "progress.jsonl", "behavior.json", "train.log"
        ]
        print("[ColabJob] 下载最终产物...", flush=True)
        for artifact in artifacts:
            try:
                result = _run_colab(
                    ["download", "-s", self._session_name,
                     f"{self._remote_run_dir}/{artifact}",
                     str(self.dir / artifact)],
                    timeout=120.0
                )
                if result.returncode == 0:
                    print(f"[ColabJob] ✅ 已下载: {artifact}", flush=True)
            except Exception as e:
                print(f"[ColabJob] 下载 {artifact} 失败: {e}", flush=True)

        # 同步到 Google Drive（通过 Colab 云端执行 Drive 挂载 + 复制）
        self._sync_to_drive()

        # 停止 Colab 会话（释放 GPU，停止计费）
        self._stop_session()

    def _sync_to_drive(self) -> None:
        """在 Colab 云端将训练结果同步到 Google Drive。"""
        drive_path = f"/content/drive/MyDrive/{DRIVE_BACKUP_FOLDER}/{self.run_name}"
        sync_script = textwrap.dedent(f"""\
            from google.colab import drive
            import shutil, os
            drive.mount('/content/drive', force_remount=False)
            os.makedirs('{drive_path}', exist_ok=True)
            src = '{self._remote_run_dir}'
            for f in os.listdir(src):
                if f.endswith(('.onnx', '.pt', '.json', '.log', '.jsonl', '.yaml')):
                    shutil.copy2(f'{{src}}/{{f}}', '{drive_path}/')
            print('Drive 同步完成: {drive_path}')
        """)

        try:
            result = _run_colab(
                ["exec", "-s", self._session_name, "-c", sync_script],
                timeout=180.0
            )
            if result.returncode == 0:
                print(f"[ColabJob] ✅ 已同步到 Drive: {drive_path}", flush=True)
            else:
                print(f"[ColabJob] ⚠️ Drive 同步失败（不影响本地产物）: "
                      f"{result.stderr.strip()[:200]}", flush=True)
        except Exception as e:
            print(f"[ColabJob] Drive 同步出错: {e}", flush=True)

    def _stop_session(self) -> None:
        """停止并释放 Colab 会话（不再计费）。"""
        if not self._session_started:
            return
        try:
            result = _run_colab(
                ["stop", "-s", self._session_name],
                timeout=30.0
            )
            if result.returncode == 0:
                print(f"[ColabJob] 会话 {self._session_name} 已停止", flush=True)
                self._session_started = False
        except Exception as e:
            print(f"[ColabJob] 停止会话失败: {e}", flush=True)

    # ------------------------------------------------------------------ 公共接口（对齐 TrainingJob）

    def poll(self) -> tuple[bool, bool]:
        """返回 (progress_changed, new_snapshot)。
        由 viz_server 的主循环调用（与 TrainingJob.poll() 相同时机），
        消费后台轮询线程写入的标志位。
        """
        # 同时检查本地 live.onnx 文件（后台轮询已经下载到此路径）
        live = self.dir / "live.onnx"
        if live.exists():
            m = live.stat().st_mtime
            if m > self._live_mtime + 0.5:
                self._live_mtime = m
                self._new_snapshot = True

        changed = self._progress_changed
        snap = self._new_snapshot
        self._progress_changed = False
        self._new_snapshot = False
        return changed, snap

    def stop(self) -> None:
        """停止训练：取消后台任务 + 停止 Colab 会话。"""
        self.status = "stopped"
        self._freeze_elapsed()
        # 取消后台轮询
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        # 停止云端会话（在线程中执行，不阻塞事件循环）
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.run_in_executor(None, self._stop_session)
            else:
                self._stop_session()
        except Exception as e:
            print(f"[ColabJob] stop() 释放会话出错: {e}", flush=True)

    def display_title(self) -> str:
        return self.behavior_title

    def stage_payload(self) -> None:
        """Colab 任务不支持课程分阶段，始终返回 None。"""
        return None

    def stage_env(self) -> dict:
        """Colab 不直接控制本地 env，返回空。"""
        return {}

    def overall_elapsed(self) -> float | None:
        if self._t0 is None:
            return None
        if self._elapsed_final is not None:
            return self._elapsed_final
        return time.time() - self._t0

    def _freeze_elapsed(self) -> None:
        if self._t0 is not None and self._elapsed_final is None:
            self._elapsed_final = time.time() - self._t0

    def payload(self) -> dict:
        """与 TrainingJob.payload() 格式对齐的响应体，多一个 backend 字段。"""
        overall_steps = int(self.progress.get("steps", 0) or 0)
        overall_total = int(self.progress.get("total", 1) or 1)
        elapsed = self.overall_elapsed()

        # 根据显卡类型匹配对应点数费率
        rate_info = next((acc for acc in COLAB_ACCELERATORS if acc["id"].lower() == self.gpu_type.lower()), None)
        rate = rate_info["rate"] if rate_info else (1.96 if self.gpu_type.upper() != "CPU" else 0.0)

        # 优先使用云端实际探测到的真实硬件型号
        display_gpu = self.actual_gpu if (self.actual_gpu and self.actual_gpu != self.gpu_type) else self.gpu_type
        if rate > 0:
            hourly_rate_str = f"约 {rate} 算力点/小时"
            cost_str = f"{round(((elapsed or 0) / 3600.0) * rate, 2)} 算力点"
        else:
            hourly_rate_str = "免费 (CPU 模式)"
            cost_str = "0 算力点 (免费)"

        gpu_title = f"{display_gpu} GPU" if self.gpu_type.upper() != "CPU" else "CPU 运行时"

        return {
            "runName": self.run_name,
            "status": self.status,
            # behavior 卡片使用简化版（Colab 模式无本地 behaviors 对象）
            "behavior": {
                "id": self.behavior_id,
                "emoji": self.behavior_emoji,
                "title": self.behavior_title,
                "description": f"Google Colab ({gpu_title}) 训练 — {self.task_id}",
                "howItLearns": "通过 Isaac Lab / MuJoCo Warp 在 GPU 上强化学习",
                "successMetric": "累计奖励提升",
                "defaultSteps": overall_total,
                "terms": [],
            },
            "progress": {
                **self.progress,
                "overallSteps": overall_steps,
                "overallTotal": overall_total,
                "overallElapsed": (None if elapsed is None else round(elapsed, 1)),
            },
            "stage": None,         # 无课程阶段
            "weights": {},         # Colab 端不暴露奖励权重
            "stageWeights": {},
            "stageSteps": [overall_total],
            "stepBudget": overall_total,
            "chosenBudget": None,
            "stageBudgets": {},
            "envs": self.num_envs,
            "helpers": 0,
            "maxHelpers": 0,
            "restarting": False,
            # ---- 额外字段（前端展示与成本看板）
            "backend": "colab",                 # 标识来自 Colab 算力
            "backendTitle": f"Google Colab ({gpu_title})",
            "hourlyRate": hourly_rate_str,
            "estimatedCost": cost_str,
            "elapsedSeconds": round(elapsed, 1) if elapsed is not None else 0,
            "session": self._session_name,      # Colab 会话名
            "taskId": self.task_id,             # microduck_rl 任务 ID
            "gpuType": self.gpu_type,           # 用户选择的卡型号 (T4/L4/A100/H100/CPU)
            "actualGpu": self.actual_gpu,       # 云端实际探测到的型号
            "driveBackup": DRIVE_BACKUP_FOLDER, # Drive 备份路径
        }

    def _write_event(self, msg: str) -> None:
        """把日志信息写到 train.log，供调试用。"""
        try:
            with open(self.dir / "train.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass
