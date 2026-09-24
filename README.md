# Microduck Lab 🦆：本地机器人学习实验室

**简体中文（默认）** · [English](README_EN.md)

本项目基于 [Jonathan Hawkins 原作者的 Microduck Lab](https://github.com/jonathanhawkins/microduck-lab)。原作提供 Mac 上的本地强化学习训练、浏览器 3D 可视化、模拟场景与机器人行为实验；这个分支在此基础上加入 **Google Colab GPU 训练入口**和**中英文切换**。它是社区实验项目，与 Pollen Robotics、Innate 和 Unitree 没有关联。

> **验证状态（2026-09-24）：**本地构建与自动化测试已通过。当时用于验证的 Colab 账号显示 **0.00 算力单位（CCU）**，因此尚未完成一次真实 GPU 训练验证。Google Drive 自动备份、断线续训、每 30 秒更新 3D 视图、保证运行 12–24 小时均未实现；Colab 的 GPU 分配和会话时长由 Google 决定。

![九只 MicroDuck 在浏览器中运行](docs/media/viewer.png)

## 相对原作者项目，增加了什么？

| 功能 | 原作者项目基础 | 本项目的升级 |
|---|---|---|
| 云端 GPU | 保留本地 CPU 训练和原有 Hugging Face 相关入口 | 新增 `/cloud` 页面，使用官方 Colab CLI 调用用户自己账号下的 GPU；先检查 CCU 余额，再启动允许的 `microduck_rl` 任务 |
| 任务与模型 | 原有本地训练、可视化和 ONNX 工作流 | 云端任务状态、单任务停止、一键停止本项目的 Colab 会话；训练结束后下载 Checkpoint 和导出的 ONNX |
| 登录方式 | 无需 Google 账号即可使用本地功能 | 仅使用 Colab 官方授权流程；本应用不接收 Google 密码或 OAuth 令牌。每位用户使用自己的账号和算力余额 |
| 语言 | 原作者界面以英文为主 | 首次打开默认中文；可切换英文，选择会在浏览器保存，并在页面间同步。导航、云端流程、训练图表和主要模拟控制已有翻译 |
| 浏览器备选方案 | — | 提供 [Colab Notebook](notebooks/microduck_train.ipynb)，Windows 用户也可在 Colab 网页中运行 |

高级面板及服务器返回的部分训练配方目前仍为英文。这里列的是**已经落地**的功能；截图中提出的 Drive 自动备份、故障自愈和训练中实时 ONNX 热加载仍属后续工作。

## 快速开始

建议使用 macOS 或 Linux，准备好 Git、Python 3.12、[`uv`](https://docs.astral.sh/uv/) 和 Node.js。安装脚本会拉取项目依赖、原作者依赖的机器人模型和参考策略：

```bash
git clone https://github.com/heranhe/microduck-lab-cloud.git
cd microduck-lab-cloud
./scripts/setup.sh
```

在一个终端启动本地实验室：

```bash
cd microduck_local
uv run duck-lab
```

在另一个终端启动浏览器界面：

```bash
cd microduck-lab-cloud/duck-viewer
npm run dev
```

打开终端显示的本地地址。实验室、本地训练和模拟页面**无需 Google 登录**。在界面顶部切换「中文 / EN」，再次访问时会保留选择。更完整的命令、性能测量和原作者功能说明见 [English guide](README_EN.md)。

## 使用 Google Colab GPU

云端训练运行的是 Pollen Robotics 官方 [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl) GPU 训练栈，而非把本地 CPU 训练代码直接搬到 Colab。Colab 的费用、GPU 供应和账号权益由 Google 管理；本项目只管理它启动的任务。

1. 在 `microduck_local` 目录安装依赖并查看账号状态：

   ```bash
   cd microduck_local
   uv sync
   uv run colab usage
   ```

2. 首次使用时，按官方 Colab CLI 给出的链接完成 Google 授权。确认命令显示的 **Current balance 大于 0**。授权凭据留在本机 CLI 中，不会提交到本仓库，也不会传给本应用。
3. 启动 `duck-lab` 和前端，打开 `/cloud` 页面。选择任务、GPU、迭代次数和并行环境数，再启动训练。
4. 在任务列表查看状态；可以停止单个任务，也可以停止本项目创建的全部 Colab 会话。成功导出后下载 `policy.onnx`。正常关闭实验室时，服务端也会尝试释放仍由本项目管理的会话。

如果余额为 0，请先确认 Colab 网页和 CLI 使用的是**同一个享有权益的 Google 账号**。即便有余额，具体 GPU 型号也可能暂时不可用。Windows 用户可在 Colab 网页打开 [训练 Notebook](notebooks/microduck_train.ipynb)；它不依赖本地 Colab CLI。

### 当前云端功能边界

- 最终 Checkpoint 和 ONNX 在训练结束后下载；训练过程中的文件不会自动同步到 Google Drive。
- 网络中断或 Colab 主动回收运行时后，任务不会自动从上一次 Checkpoint 接续。
- 3D 界面目前不支持每 30 秒加载云端临时 ONNX；也不保证 12–24 小时持续运行。
- 云端导出的策略在用于真实机器人前，仍需按原项目的仿真到现实流程评估。

## 原作者项目保留的能力

- **本地强化学习：**在普通 Apple Silicon Mac 上用 MuJoCo 和 Stable Baselines 3 迭代策略，并导出 ONNX。
- **3D 可视化：**同屏观察多个策略、回放训练结果、录制截图、视频和 GIF。
- **行为教学：**在网页中训练和观察新动作；关键帧动画与 IK 编辑器可辅助构造动作。
- **模拟世界 `/sim`：**房间、传感器、角色行为、玩具整理与足球场景；支持编辑场景和记录回放。
- **训练分析 `/train`：**查看奖励、回合长度和不同训练运行的差异。
- **多机器人：**MicroDuck、Unitree G1 和 Innate MARS 共用实验室的可视化框架。

原作者对实验结果、物理模型和真实机器人部署有大量重要说明。需要深入训练时，请阅读 [完整英文说明](README_EN.md)、[项目约定](AGENTS.md) 和 [`microduck_local` 训练手册](microduck_local/README.md)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `microduck_local/` | 本地训练、`duck-lab` 服务端、Colab 任务管理 |
| `duck-viewer/` | Next.js 浏览器界面和语言切换 |
| `notebooks/` | Colab 网页训练备选方案 |
| `docs/` | 原作者的实验记录、路线图和媒体素材 |
| `scripts/setup.sh` | 工作区依赖安装脚本 |

## 来源与许可

本项目沿用原作者项目及其 [Apache-2.0 许可证](LICENSE)。原作者仓库：[jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab)。本分支新增功能的范围见上方对照表；代码变更可在 Git 历史中逐项查看。
