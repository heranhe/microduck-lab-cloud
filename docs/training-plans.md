# 训练方案使用说明

在实验室顶部点击“训练方案 · 创建 / 导入”，或打开 `/plans`。

## 创建与使用

1. 在“已有模板”选择本地配方或官方云端任务；也可以导入方案文件。
2. 填写名称、目标和成功标准，配置本地或云端实现。
3. 保存方案。页面会显示实际文件路径和版本，支持另存为、导出和历史版本。
4. 选择执行位置，点击“开始训练”。本地任务到实验室观察与停止，云端任务到云端页面管理。

本地使用 MuJoCo + SB3 PPO，官方云端使用 MuJoCo Warp + mjlab。两套实现分别配置，不会自动转换本地奖励、课程或参考动作。本地“保持站立”与云端“行走与跌倒恢复”是分别命名的任务。

## 文件导入

- 训练方案：`.json`，或包含 `TRAINING_PLAN = {...}` 字面量的 `.py`。先从页面导出模板，再修改与导入。
- 动作片段：使用单独的“导入动作 JSON”入口，文件须符合当前机器人关节数量和关键帧格式。
- `.onnx` 与检查点是模型成果，不属于训练方案。
- 任意 Python 训练脚本不能直接执行；需要开发者适配到已有配方/任务注册机制。Python 导入只读取声明式配置，不执行文件代码。
- JSON/YAML 并非任意通用训练格式。当前导入支持本项目规范的 JSON 与 Python 字面量，不支持 YAML。

方案字段示例：

```json
{
  "version": 1,
  "title": "单脚站立实验",
  "description": "减少单脚站立时的身体晃动",
  "success": "在多个初始姿态下稳定站立，并观察实际动作",
  "robot": "microduck",
  "source": "manual",
  "local": {
    "behavior": "one_leg",
    "steps": 2500000,
    "weights": {"smooth_moves": 3.0},
    "stageSteps": {},
    "stageWeights": {}
  },
  "cloud": null
}
```

保存时补齐基础奖励权重，避免继承其他任务的临时调参。本地总预算范围与现有训练服务一致，为 100,000–40,000,000 步。训练器按 rollout 批次采样，实际采样步数可能略超过预算。

云端配置支持 `task`、`iterations`、`envs`、`gpu`（Colab）、`flavor`（Hugging Face）。目前接入 `Mjlab-Velocity-Flat-MicroDuck` 和 `Mjlab-VelStand-Flat-MicroDuck`。VelStand 需要官方教师检查点；启动前会检查。账号连接、硬件可用性与最终提交仍由现有云端服务验证。

## 关键帧

在“动画”窗口调整姿态、添加关键帧并保存，点击“创建训练方案”。方案页绑定动作与对应机器人的模仿配方。导入动作文件也会生成模仿草稿。

参考动作会嵌入方案版本。训练时生成独立版本的片段快照。方案页可以预览和导出参考动作；动画预览与学习结果分别验证。当前没有官方 GPU 关键帧模仿适配，因此这类方案只能选择本地。

## 保存与版本

默认目录为 `microduck_local/training-plans/<id>/v<revision>.json`，可由 `MICRODUCK_PLANS_DIR` 修改。页面显示后端实际路径，不依赖写死的本机目录。

每次保存创建新的不可变版本；同时更新 `plan.json`。旧版本可查看、导出或另存为新方案。并发修改会返回版本冲突，要求重新加载。每次提交生成 `launch-*.json`，记录完整方案快照、执行位置与任务。本地 run 目录额外保存 `training-plan.json`。

动作目录由后端 `MICRODUCK_CLIPS_DIR` / runs 相邻目录决定。文件上传成功后可以查看目录、复制路径及导出。训练窗口完成任务后支持将调整后的配方“保存为我的方案”。

## AI 辅助创建

在“AI 创建”中配置 OpenAI 兼容模型服务的基础地址、模型名及密钥，也支持本机兼容服务。AI 根据已有配方和任务生成可编辑草稿，不会自动启动训练。草稿必须经同样的后端校验，再由用户检查、保存与执行。

新环境、新奖励函数和官方 GPU 模仿任务仍需开发接入；AI 不能凭一句话让不存在的训练实现变得可用。

凭据仅保存到训练服务端 `training-plans/ai-settings.json`，文件权限 0600，接口不返回密钥，目录已 gitignore。配置也可通过 `MICRODUCK_AI_BASE_URL`、`MICRODUCK_AI_MODEL`、`MICRODUCK_AI_API_KEY` 提供。远程服务会收到用户目标与模板说明，按服务商规则计费。

## 验证

前端：`npm test`、`npm run build`。

后端：`PYTHONPATH=src uv run --with pytest python -m pytest tests/test_training_plans.py tests/test_lab.py tests/test_clips.py tests/test_colab_jobs.py tests/test_hf_jobs.py`。

界面测试建议使用独立的 plans、runs、clips 和 lab-state 目录，避免修改已有数据；云端提交使用模拟服务验证，不需要分配付费 GPU。
