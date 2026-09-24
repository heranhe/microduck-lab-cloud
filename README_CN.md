<div align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</div>

# Microduck Lab Cloud 🦆☁️

面向 [Microduck](https://pollen-robotics.com/microduck) 双足机器人的高阶强化学习实验、云端训练工作流与交互式 3D 仿真套件。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)

> **Microduck Lab Cloud** 在 [microduck-lab](https://github.com/jonathanhawkins/microduck-lab) 原生架构的基础上进行了深度拓展，新增了一系列高敏捷度 RL 行为动作库（白鹤亮翅、单腿跳跃、90°跳跃转身、爆发远跳）、支持中英双语切换的 Web 3D 实时仿真交互界面、基于自动化目标搜索的课程学习系统，以及 Google Colab / 云端 GPU 远程训练支持。

### 🆚 相比原版（microduck-lab）的核心演进与区别

| 核心维度 | 原版 `microduck-lab` | 本项目 `microduck-lab-cloud` 增强版 |
| :--- | :--- | :--- |
| **🌐 语言支持** | 仅纯英文界面与文档 | **全栈中英双语支持**：Web 3D 界面、遥测控制面板、教学技巧菜谱及完整文档支持一键中英文即时切换。 |
| **☁️ 云端算力调用** | 仅支持本地 Mac CPU / 基础 MPS，无法处理大规模并行或超深层策略网络 | **集成 Google Colab / Google AI 算力调度引擎**：可直接调用 **NVIDIA T4、L4、A100、H100** 等高端芯片算力，突破本地散热与性能天花板。 |
| **⏱️ 训练时长保障** | 受限于本地笔记本续航与发热，通常仅用于几分钟的原型速测 | 单任务支持长达 **12 ~ 24 小时**的稳定超长周期强化学习训练，配备多阶段 Checkpoint 保存与故障自愈。 |
| **🚨 算力防扣费熔断** | 无云端实例生命周期管理 | **一键安全熔断（Kill-Switch）**：前端 HUD 专设安全熔断开关，遭遇异常或死锁时，一键即可强制销毁并释放所有计费 GPU 实例，彻底杜绝扣费失控。 |
| **📡 远端实时反向流** | 无 | **30 秒实时热流加载**：远端训练时每 30 秒自动回传 `live.onnx` 快照并在本地 3D 视口实时更新动作；训练完成自动双重归档至 Google Drive。 |
| **🥋 敏捷动作扩展** | 基础行走、小跳与后空翻 | **新增敏捷高难动作库**：白鹤亮翅（单腿悬停平衡）、单腿连续跳、90°腾空跳跃转身、高爆发向前远跳及配套动作 Clips。 |

无需专业 CUDA GPU，不仅可在 **Apple Silicon Mac 本地** 运行，更能一键调度 **云端 GPU 实例**，高效训练 ~25 cm 开源双足机器人 Microduck 的强化学习控制策略，并在浏览器中实时观察每个智能体的行走、学习过程与敏捷动作。

![Duck Lab 交互式查看器: 9只小鸭机器人实时仿真](docs/media/viewer.png)

| 本地 CPU 训练步态（结合 BAM 真实舵机动力学） | 后空翻展示：辅助弹射、策略着陆与手持交接 |
|---|---|
| ![running](docs/media/running.gif) | ![backflip](docs/media/backflip.gif) |

官方的 [microduck_rl](https://github.com/pollen-robotics/microduck_rl) 依赖 MuJoCo Warp 并且必须配备高端 CUDA GPU。而本项目旨在利用你手头现有的笔记本电脑或云端服务器快速构建原型。它与官方完全共享同一个 MJCF 机器人几何与动力学模型、相同的 **61 维观测 (Observations) / 14 维动作 (Actions)** 部署接口契约，以及标准的 50 Hz 控制频率。在本项目中设计的全新行为，可以直接无缝迁移到官方 Sim2Real 流程中，导出的 ONNX 模型也可直接被官方工具链加载使用。

*声明：本项目为独立开源衍生项目，不隶属于 Pollen Robotics。*

---

## 核心特性与目录结构

- **`microduck_local/`**：基于 CPU-MuJoCo + Stable Baselines 3 PPO 的高性能训练框架
  - `train-walk` / `train-behavior`：速度控制行走训练与丰富的可教导技巧动作库，集成 mjlab 蒸馏奖励、动作对称性增强、观测标准化与惩罚符号守卫
  - **✨ 新增敏捷行为动作库 (Agile Behaviors)**：
    - 🕊️ `white_crane`（白鹤亮翅）：动态单腿离地平衡、抬腿悬停与高稳定性上身姿态控制
    - 🦿 `single_leg_hop`（单腿跳跃）：连续动态单腿跳动与触地缓冲自适应控制
    - 🔄 `jump_turn`（跳跃转身）：腾空 90 度偏航旋转并在触地瞬间平稳消解角动量
    - 🦘 `long_jump`（向前远跳）：高爆发前向腾跃与安全抗冲击受力着陆
  - **☁️ 云端训练与自动化管线**：
    - `colab_runner.py`：支持在 Google Colab 或远程 Linux 服务器上一键启动训练
    - `goal_training.py` & `search_hop.py`：目标导向的自动化多阶段课程超参数搜索与奖励函数调优
  - **两套舵机执行器动力学模型**：
    - 极速线性化 XML 舵机模型
    - 高保真 BAM XL330 电压饱和动力学模型（经 Numba JIT 加速），针对高爆发及饱和极限动作
  - `export-walk`：将策略导出为部署级 ONNX 模型（内置 Observation Normalizer 归一化层）
  - `eval-walk`：无头自动化评估电池（跌倒率、指令跟踪误差）
  - `render-rollout`：生成高清 MP4 视频，以及**携带每帧身体高度、倾角和足底接触力标签、可直接供 AI 编码助手读取的接触表 (Contact Sheet)**（[示例](docs/media/contact-sheet.png)）
  - `bench-walk` / `bench-envs`：自动基准测试，为你当前机器匹配最高吞吐量的并行环境数
  - `duck-lab`：驱动前端浏览器 3D 查看器的 WebSocket 流式通讯后端
- **`duck-viewer/`**：基于 Next.js 15 + React Three Fiber 的现代化 3D 查看器
  - **🌐 中英双语一键切换**：支持在中英文界面间自由切换，涵盖遥测仪表盘、控制面板和技巧菜谱
  - 多机器人同台仿真：通过 25 Hz WebSocket 实时同屏渲染多只 Microduck，支持拖拽策略芯片直接在步态中热替换“大脑”
  - **🎓 教学面板 (Teach Panel)**：通过自然指令或关键词选择预设技巧（例如“单腿站立”），查看纯英文/中文描述的奖励项构成；每 ~15 秒快照热加载，支持直接拖动滑块实时调节奖励权重，无需修改 Python 代码
  - 多阶段复杂技能编排：对于硬核动作（如 5 阶段级联后空翻），界面实时解说并展示课程学习链条
  - **🎬 动作面板 (Animate Panel)**：内置关键帧姿态编辑器与动画控制 Rig。在浏览器中可视化编辑动作片段，点击“训练此动作”即可通过强化学习使其物理可执行
  - **🎥 捕获面板 (Capture Panel)**：📷 一键导出无干扰高清 PNG 截图；🎥 电影级运镜跟踪录制，自动生成高质量 MP4 和 GIF 动图，方便直接贴入 Issue 或 PR
  - **⤓ 一键下载 ONNX**：直接下载包含归一化器的完整端到端神经网络权重
  - **⚙️ 设置与 Hugging Face (BYOK)**：支持配置个人的 Hugging Face Token，方便无缝对接云端模型仓库

![在浏览器中教学一个动作](docs/media/teach.png)

---

## 快速上手 (Quick Start)

### 环境准备
* 支持操作系统：macOS（Apple Silicon M 系列推荐）或 Linux
* 必备工具：[uv](https://docs.astral.sh/uv/)（现代高性能 Python 包管理工具）、Node.js 20+
* 磁盘空间：约 3 GB 用于依赖项及官方模型检出

```bash
# 1. 克隆本项目并进入目录
git clone https://github.com/heranhe/microduck-lab-cloud.git && cd microduck-lab-cloud

# 2. 检出官方依赖项（用于模型文件与官方对比）
git clone https://github.com/pollen-robotics/microduck
git clone https://github.com/pollen-robotics/microduck_rl

# 3. 初始化 Python 环境与单元测试
cd microduck_local
uv sync
uv run --with pytest pytest tests/   # 运行接口契约与动作策略测试，确认全部通过

# 4. 训练你的第一个行走策略 (在 M 系列 Mac 上仅需数分钟)
uv run train-walk --envs 32 --steps 3_000_000 --run-name first-gait
uv run export-walk runs/first-gait
uv run eval-walk runs/first-gait/policy.onnx

# 5. 启动流式后端与 Web 3D 查看器
uv run duck-lab runs/first-gait ../microduck/policies/alpha_walking.onnx

# 打开另一个终端启动前端
cd ../duck-viewer
npm install
npm run dev   # 访问终端打印的 http://localhost:3000
```

打开浏览器后，你可以点击右上角的语言切换按钮选择 **简体中文**，在 🎓 教学面板中输入“单腿站立”或选择“白鹤亮翅”观察训练进展。

---

## 性能表现 (在 Apple M 系列芯片实测)

详细测试方案与所有对比实验可参考 [`microduck_local/README.md`](microduck_local/README.md)：

- **~16,500 仿真步/秒 (env-steps/s)**：在默认 32 环境配置下，每训练 100 万步仅需约 1 分钟；峰值吞吐量可达 27k 步/秒
- **共享编译 MuJoCo 模型**：采用 fork + 写时复制 (Copy-on-Write) 机制，64 并行环境的内存占用从 **41 GB 直降到 1.5 GB**
- **低延迟 IPC 通信**：使用基于信号量 (Semaphore) + 共享内存的 IPC，替代传统的 pipe/pickle 序列化开销
- **Numba JIT 融合 BAM 驱动模型**：实现与 Numpy 参考实现逐位精度完全一致的高性能物理执行器模拟
- **自适应 MPS 硬件加速**：智能判定在有吞吐增益的阶段自动启用 Apple Silicon GPU (MPS) 进行 PPO 权重更新

---

## 训练自定义技能与动作 (Teach Panel)

所有技能动作都以明晰的代码化奖励配方组织在 [`microduck_local/src/microduck_local/behaviors/`](microduck_local/src/microduck_local/behaviors) 中。一个 `Behavior` 包含：
1. 一组清晰可解释的奖励项（Reward Terms）
2. 触发关键词
3. （可选）针对高难度动作的多阶段课程调度（Curriculum）

只需在 `behaviors/` 中新增一个行为并补齐单元测试，启动系统后即可在 Web 查看器的教学面板中即时加载该技能，并使用动态滑块调整权重。详见 [`microduck_local/AGENTS.md`](microduck_local/AGENTS.md)。

---

## 关键帧动作编辑与模仿学习 (Animate Panel)

在 🎬 动画编辑面板中，你可以通过可视化的方式设计动画轨迹：

![拖拽调节姿态控制柄](docs/media/animate-rig.gif)

- **直观姿态拖拽**：直接点击机器人身体部位拖动，或使用特制的 **🎮 控制骨架 (Control Rig)**（`squat` 下蹲、`lean` 俯仰、`sway` 侧摆、`stance` 开合、`twist` 扭腰等）。控制轴正交设计，确保调节单一关节时不破坏机器人的双足接地基础。
- **时间轴与关键帧**：支持自动打帧、时间轴快进/回放与循环动画；所有姿态由后端物理求解器保证预览真实。
- **⚡ 训练此动作 (DeepMimic 模仿学习)**：将动画片段作为奖励参考。利用模仿学习，将像“探索后空翻”这种极具难度的全空间盲目搜索转变成易于收敛的轨迹跟踪问题，且依然完全兼容 61 维部署契约。参考 [`motion.py`](microduck_local/src/microduck_local/motion.py)。

---

## 捕获与分享 (Capture Panel)

查看器内置强大的录屏与截图工具，无需第三方录屏软件：

![捕获面板：支持导出 MP4 视频与 GIF 动图](docs/media/capture.png)

- **📷 截图 (Shot)**：一键抓取全分辨率清晰 PNG，自动隐去高亮选择光圈，便于撰写技术报告。
- **🎥 运镜录像 (Record)**：智能摄像机自动平滑滑行到小鸭前方 ¾ 视角，以微弱缓动保持视觉张力，同时使用 WebGL 画布录制流捕获纯净画面（界面 UI 与 DOM 悬浮层不会遮挡画面）。
- **自动转码**：录制完成上传至后端服务自动转码为高质量 H.264 **MP4** 以及调色板优化的 **GIF** 动图，一键下载直接贴入 GitHub PR。

---

## 导出与部署：ONNX & Hugging Face

- **⤓ 一键下载策略 (Exported ONNX)**：在策略列表中悬停即可下载对应策略的 `.onnx`。所有导出均已内嵌观测归一化层（Observation Normalizer），避免因缺少归一化参数导致策略失效。即使训练仍在进行中，也可直接抓取最新实时的 `live.onnx` 快照。
- **🤗 连接 Hugging Face (BYOK)**：点击齿轮按钮可填入自己的 Hugging Face Token（Token 仅保存在本地 `hf-token.json` 并受严格权限保护，浏览器端永远无法探查密钥），方便后续与云端模型仓库和算力任务对接。

---

## 关于 Sim2Real 现实迁移

本项目的主要目标是为强化学习实验提供**分钟级的敏捷验证闭环**（快速试错奖励函数、探索新动作课程）。为了保证在普通 Mac 笔记本上的极速运行，本框架运行的是官方域随机化（Domain Randomization）的一个精简子集。

当你在本项目中验证某一动作切实可行后，只需将该环境参数迁移至 `microduck_rl` 的 mjlab 配置中，即可在高端 GPU 上启动完整的实机向 Sim2Real 域随机化迁移训练。两边遵循完全一致的动作与观测接口契约，因此代码迁移是完全机械化、无缝平移的。

---

## 致谢与上游开源项目

- **[microduck-lab](https://github.com/jonathanhawkins/microduck-lab)**（作者：Jonathan Hawkins）：提供出色的 CPU-MuJoCo 训练基底与 Next.js 交互式架构原型。
- **[microduck](https://github.com/pollen-robotics/microduck)** 与 **[microduck_rl](https://github.com/pollen-robotics/microduck_rl)**（作者：Pollen Robotics）：提供开创性的 Microduck 双足机器人硬件设计、MJCF 模型与官方强化学习工具栈。

---

## 开源许可证 (License)

本项目基于 [Apache License 2.0](LICENSE) 开源（与上游 Microduck 官方生态保持一致）。  
本项目不隶属于 Pollen Robotics，亦未获得其官方背书。"Microduck" 是 Pollen Robotics 的商标。
