# Colab VelStand 教师模型配置

VelStand 需要两份原版 PyTorch Checkpoint。已有 ONNX 策略不能代替含训练状态的 Checkpoint。教师模型未准备好时，后端会在分配 GPU 前拒绝提交并列出缺失文件。

从你有访问权限的 W&B 项目获取：

| 作用 | W&B run | 文件 | 默认本地位置（相对 microduck_local） |
| --- | --- | --- | --- |
| 站起教师 | pollen-robotics/mjlab_microduck/69u48n8l | model_9750.pt | teachers/69u48n8l/model_9750.pt |
| 行走教师 | pollen-robotics/mjlab_microduck/441tzs6d | model_3750.pt | teachers/441tzs6d/model_3750.pt |

也可以在启动后端前设置 `MICRODUCK_STAND_TEACHER` 和 `MICRODUCK_WALK_TEACHER` 为对应文件的绝对路径。默认目录添加文件后无需重启即可重新提交。

后端在分配成功后上传这两份模型，训练与导出均从上传路径读取，不需要把 W&B API Key 传到 Colab。模型不存在或为空会被提前拦截；Checkpoint 的结构和网络参数兼容性仍由上游加载器检查。

远端源码固定到 `cb70b792312d559a4da09064d92009079671815f`，使教师配置与训练代码可追溯。没有禁用专家模仿项，也没有把 VelStand 换成其他训练任务。

当前验证：本地编排回归测试通过；缺模型提交会在 GPU 分配前失败。尚未拿到真实教师 Checkpoint，因此上传后训练、导出与下载全流程仍待真实验证。
