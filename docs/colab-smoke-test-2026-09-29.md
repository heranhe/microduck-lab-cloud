# Colab T4 冒烟测试

测试日期：2026-09-29。任务 ID：`30f57574ae24`。

## 配置与结果

官方任务 `Mjlab-VelStand-Flat-MicroDuck`，T4，16 个环境，1 次迭代。

| 阶段 | 结果 |
| --- | --- |
| 账号授权 | 成功 |
| 云端分配 | 成功：Tesla T4，15 GB |
| 依赖安装 | 完成，远端状态从 setup 进入 training |
| 训练启动 | 成功，日志出现 mjlab 机器人初始化信息 |
| 训练完成 | 未确认 |
| ONNX 导出与下载 | 未完成 |
| 自动释放 | 后端 released=true，随后查询 Google 无活动会话 |

## 失败原因

本地编排器调用 `colab exec` 查询远端状态超过 30 秒，抛出 `subprocess.TimeoutExpired`。异常进入整个任务的失败与释放流程，导致运行时被提前终止。

单次进度查询超时不能证明远端训练失败。当前轮询只对 CLI 非零退出码继续重试，对超时异常则直接结束任务；首次 GPU 初始化期间，这一行为会中断训练。

本次结果为冒烟测试未通过：已验证真实云端连接和训练启动，尚未验证 Checkpoint、ONNX 与下载链路。建议对进度查询设置有限次数重试、最后成功心跳和阶段超时，再重复相同的小规模测试。

## 修复与复测

已将进度查询超时设置为 60 秒，超时、非零退出码、无有效状态均计入连续失败次数；成功读取状态后清零并更新最后心跳。单次失败保持运行并提示重试，连续 6 次失败才结束任务并尝试释放。取消请求仍优先处理。15 项单元测试通过，覆盖超时后恢复及达到重试上限后的释放。

相同配置复测任务：`760d06a7c68d`。获得 Tesla T4 / 15 GB，查询正常，无提前超时终止。远端初始化 GPU 仿真和模型后真实报错：`wandb.errors.errors.UsageError: No API key configured`。VelStand 的 PPO 蒸馏实现通过 W&B 下载教师 Checkpoint，`WANDB_MODE=disabled` 不会取消这一依赖。错误日志和训练日志已下载，运行时已自动释放，Google 查询无活动会话。

结论：进度查询超时处理已修复；完整训练仍未通过，下一项阻塞是官方 VelStand 的教师模型获取凭据。需配置合法模型访问方式后再次验收 ONNX 导出与下载。

## 再次复测

任务 `4f00b66bc6e9`，相同配置（T4、16 环境、1 次迭代），约 4 分 50 秒后结束。真实设备为 Tesla T4 / 15 GB。依赖安装、GPU 初始化成功，进度查询没有失败；在训练算法初始化时仍因 W&B 未配置 API Key，无法获取教师 Checkpoint 而退出，未完成训练迭代和 ONNX 导出。

`train.log`、`error.log` 已下载至 `microduck_local/runs/colab/4f00b66bc6e9/`。任务 `released=true`，随后 Google CLI 查询确认无活动会话。完整冒烟测试仍未通过，阻塞与上轮一致。

## 官方 Velocity 端到端复测

任务 `5e4f52d386c2`，官方 `Mjlab-Velocity-Flat-MicroDuck`，免费层 T4，16 环境，1 次迭代。远端设备确认为 Tesla T4 / 15 GB；状态按 `setup → training → exporting → done` 推进。训练日志记录了第 1 次迭代、损失与奖励指标。Checkpoint（4.6 MB）及 ONNX 策略（775 KB）均已下载到 `microduck_local/runs/colab/5e4f52d386c2/`。ONNX 官方 checker 通过，模型输入 `obs[1,61]`、输出 `actions[1,14]`。任务状态 `done`、`released=true`，轮询失败数为 0。

这证明当前 Google 账户在本次测试中确实连接并使用了 Colab 云端 T4，而不是本机 CPU。1 次迭代只用于链路验收，不能证明策略已经学会行走。

网页增加单独的“官方行走”动作入口，映射到已通过冒烟的 Velocity 任务。它与本地动作配方及 VelStand 明确区分。VelStand 在缺少官方两份教师 Checkpoint 时会在分配 GPU 前返回明确的 503 提示；它的端到端训练仍待取得有权限的模型文件后复测。
