# 更新日志

本项目以 `main` 分支提供源码更新；每次重要修改在这里记录，不再为日常更新创建 GitHub Release。已有标签保留，方便追溯历史代码。

## 2026-09-30 · 训练方案编辑与参考动作预览修复

1. 修复保存响应覆盖等待期间新增编辑的问题；保留新输入，并关联正确的方案 ID 与已保存版本。新增编辑仍标记为未保存，需再次保存后才能训练。
2. 修复动作、历史版本和 AI 草稿异步加载后使用过期修改状态的问题；切换时依据当前草稿确认，取消后保留编辑。
3. 修复仅带 `plan`、缺少 `revision` 的动作预览链接；默认加载最新版本，非法版本号提示错误，不再请求空动作名。
4. 增加延迟响应、连续编辑、另存为、切换方案和参数边界的回归测试。
5. 将 GitHub Release 的历史更新内容迁移到本文件；删除 Release 页面，保留提交与标签。

验证：前端 354 项测试、ESLint 和生产构建（含 TypeScript）通过；方案与实验室后端回归测试 142 项通过。

升级：更新 `main`，在 `duck-viewer` 中安装依赖并重启前端。重启 `duck-lab` 以加载最新方案读取接口；若有训练正在运行，待完成或保存停止后再重启。无需修改训练模型或已有方案数据。

---

## 历史版本说明

以下内容保留原 Release 文案；其中的 CI 状态、验证数量和限制描述属于当时记录。旧 Release 链接改为标签代码链接。


## 2026-09-30 · v1.2.0

v1.2.0 · 实时训练监看与可复用训练方案

[查看历史代码](https://github.com/heranhe/microduck-lab-cloud/tree/v1.2.0)

## v1.2.0 — 实时训练监看与可复用训练方案

本次更新解决训练进度、得分与机器人动作分散在不同位置，收起面板后难以持续观察的问题。

### 更新内容

- 顶部常驻训练状态卡：任务、累计步数、完成比例、耗时、预计剩余时间、动态得分曲线与最新得分；可保存检查点后停止。
- 得分曲线与详细训练面板共用绘图和采样逻辑；新任务清空旧曲线，课程阶段使用累计步数，重启回退时移除旧尾部。
- 场景学员随训练检查点更新动作，详细面板显示奖励明细与阶段，便于比较得分变化与实际表现。
- 新增 `/plans`：模板创建、声明式 JSON/Python 方案导入、关键帧动作导入、历史版本、导出、参考动作预览和执行快照。
- 训练配方可保存为方案；动作关键帧可转成模仿训练方案。参考动作嵌入版本，方案保存采用版本冲突检查。
- AI 辅助生成可编辑方案草稿，使用用户配置的兼容服务；草稿不会自动启动训练。
- 区分本地配方与官方 GPU 任务，避免将本地“保持站立”误映射为云端官方任务。
- 整理鸭子训练场、游乐场、录制、动作关键帧编辑与训练方案入口。
- 中英文 README 前部加入训练监看截图、说明和使用入口。

### 使用与升级

更新到 `main` 或检出 `v1.2.0`，参考 README 的快速开始安装依赖，然后重启 `duck-lab` 和前端。在实验室开始本地训练即可查看常驻进度和实时得分；打开 `/plans` 创建或导入方案。

### 验证与限制

前端 335 项测试通过，构建/TypeScript 与 ESLint 已验证；后端训练方案、实验室、动作及云端任务 220 项回归测试通过，详见随版本提交的验证记录。

得分曲线采集当前页面打开期间的数据，刷新后重新采集；云端任务暂未提供同等实时得分监看。真实云 GPU 提交和外部 AI 服务未在此次验证中调用；关键帧模仿目前仅支持本地。Python 导入只读取声明式配置，不执行任意脚本。

此前后端全量验证记录包含一个未解决的足球争球测试失败，详见 `docs/training-plans-validation.md`；不能据此声称全部仿真测试通过。截图和得分不代表已学会稳定动作，也不代表硬件部署验证。

### GitHub 检查状态（发布时）

发布 PR 的前端检查与 Windows 后端检查已通过，Linux/macOS 全量仿真检查仍在运行，尚未确认通过。查看 [本次 CI](https://github.com/heranhe/microduck-lab-cloud/actions/runs/36664783730) 获取最终结果。


## 2026-09-29 · v1.1.0

v1.1.0 · Worlds, Soccer & Cloud Training / 世界模拟、足球与云端训练

[查看历史代码](https://github.com/heranhe/microduck-lab-cloud/tree/v1.1.0)

## 简体中文

本发行版对应 `main` 的提交 **08487ef3934969ebe80ded040ec318390b3f1ed5**，包含截至本次发布的最新已提交代码与中英文文档。页面下方的 **Source code (zip / tar.gz)** 即为此版本源码。

### 主要内容

- **世界模拟与机器人足球**：包含原项目的客厅、玩具房、人物跟随、1v1 / 2v2 / 3v3 足球、传感器可视化、场景编辑和时间轴回放。
- **动作编辑与多机器人**：保留关键帧、IK、模仿训练，以及 MicroDuck、Unitree G1 和 Innate MARS 的实验框架。
- **统一训练流程**：在教学面板选择动作和本机 / Colab / Hugging Face 算力；增加任务状态展示、本地保存并停止、从检查点继续训练，以及云端任务取消与释放处理。
- **Colab GPU**：免费层 T4 请求不再以付费计算单元余额为前提；兼容性检查保留用户选择的动作。改善轮询超时重试、运行时释放和结果下载。
- **中英文界面与介绍**：完善主要界面翻译；扩充两种语言的 README，加入世界场景、足球、动作编辑、训练分析的截图与动图，并注明原项目来源。

### 安装与更新

已有 Git 克隆：先保存自己的修改，再运行 `git pull --ff-only origin main` 获取当前主分支，或检出 `v1.1.0` 固定使用本发行版。新用户可下载下方源码包，解压后在项目根目录运行 `./scripts/setup.sh`，再按 README 启动后端与前端。安装需要联网获取依赖、机器人模型和参考策略；这是源码发行版，不是独立安装程序。

### 验证与已知限制

- 前端 lint、生产构建及 329 项测试通过；文档图片、相对链接和导航锚点检查通过。
- Python 全量测试：1,766 通过、377 跳过、1 失败。`test_with_nobody_to_contest_it_we_touch_first` 的先触球结果为 13/16，低于 85% 要求；在本轮合并前的 `main` 快照中也复现，尚未修复。
- 官方 Velocity 任务已在真实免费层 T4 上完成 1 次迭代、Checkpoint / ONNX 下载与运行时释放。这验证了训练链路，不代表策略已学会行走。VelStand 仍需合法获取的教师模型，端到端验收尚未完成。
- 云端自动恢复、Google Drive 自动备份和训练中实时 ONNX 预览尚未实现；GPU 分配、费用与会话期限由服务商决定。
- 本地仿真用于原型实验，实机部署仍需官方训练与验证流程。

## English

This release pins **08487ef3934969ebe80ded040ec318390b3f1ed5** from `main`, including the latest committed code and bilingual documentation at publication. The **Source code (zip / tar.gz)** downloads below contain this exact version.

### Highlights

- **Simulated worlds and soccer:** upstream living rooms, playrooms, person following, 1v1 / 2v2 / 3v3 soccer, sensor visualization, scenario editing and timeline replay.
- **Motion authoring and robots:** keyframes, IK, imitation training, and the MicroDuck, Unitree G1 and Innate MARS experimental framework.
- **Unified training:** choose the action and local / Colab / Hugging Face compute in the teaching panel; persistent task state, local save-and-stop and checkpoint continuation, plus cloud cancellation and runtime cleanup.
- **Colab GPU workflow:** free-tier T4 requests no longer require a paid compute-unit balance. Task compatibility preserves the selected action; polling retries, cleanup and result downloads are improved.
- **Chinese and English:** expanded UI translations and README guides with world, soccer, motion and training-analysis galleries, with attribution to the original project.

### Install or update

For an existing clone, save local changes before running `git pull --ff-only origin main`, or check out `v1.1.0` to pin this release. New users can download and extract the source archive below, run `./scripts/setup.sh` from its root, and follow the README to start the backend and viewer. Setup downloads dependencies, robot models and reference policies. This is a source release, not a standalone installer.

### Validation and known limitations

- Frontend lint, production build and all 329 frontend tests passed. Documentation image paths, relative links and navigation anchors passed checks.
- Full Python suite: 1,766 passed, 377 skipped, 1 failed. `test_with_nobody_to_contest_it_we_touch_first` resolves 13/16 episodes, below its 85% threshold. The same failure was reproduced on the pre-merge `main` snapshot and remains unresolved.
- The official Velocity task completed one iteration on a real free-tier T4, downloaded its checkpoint and ONNX, and released the runtime. This validates the workflow, not walking quality. VelStand still requires authorized teacher models and has not passed end-to-end acceptance.
- Automatic cloud recovery, Drive backup and live cloud ONNX previews are not implemented. Providers control GPU allocation, pricing and session lifetime.
- Simulation results are for prototyping; hardware deployment requires the official training and validation process.

Worlds, soccer, animation and the multi-robot foundation come from [Jonathan Hawkins' Microduck Lab](https://github.com/jonathanhawkins/microduck-lab). See the [Chinese README](https://github.com/heranhe/microduck-lab-cloud/blob/v1.1.0/README.md) or [English guide](https://github.com/heranhe/microduck-lab-cloud/blob/v1.1.0/README_EN.md) for details. Apache-2.0.


## 2026-09-24 · v1.0.0

v1.0.0: Microduck Lab Cloud Initial Release

[查看历史代码](https://github.com/heranhe/microduck-lab-cloud/tree/v1.0.0)

### Initial Release of Microduck Lab Cloud 🦆☁️

#### Highlights
- **Expanded Agile Behaviors**: Added imitation and reinforcement learning pipelines for White Crane posture, Single-Leg Hop, 90-degree Jump Turn, and Long Jump.
- **Interactive Multi-language Viewer**: Added bilingual (English / 简体中文) UI support, real-time 3D simulation streaming, and teach panel controls.
- **Cloud & Remote Training Ready**: Added Google Colab runner scripts and automated curriculum goal training.
- **Robust Verification**: Comprehensive contract and behavior test suites with pytest.
- **Open-source & Upstream Compliant**: Apache 2.0 licensed, built upon microduck-lab and official Pollen Robotics stacks.
