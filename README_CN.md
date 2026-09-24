# Microduck Lab 🦆

这是基于[原作者 Microduck Lab](https://github.com/jonathanhawkins/microduck-lab) 的实验项目：在本地观看和训练 MicroDuck，并可选择用自己的 Google Colab 算力运行官方 `microduck_rl` GPU 训练。完整技术说明见 [README.md](README.md)。

## 本地启动

在 macOS 或 Linux 上，先安装 `git`、Python 3.12、[`uv`](https://docs.astral.sh/uv/) 和 Node.js，然后运行：

```bash
git clone https://github.com/heranhe/microduck-lab-cloud.git
cd microduck-lab-cloud
./scripts/setup.sh
cd microduck_local
uv run duck-lab
```

另开一个终端：

```bash
cd microduck-lab-cloud/duck-viewer
npm run dev
```

打开终端显示的本地网页地址。顶部可切换中文和英文；语言设置会保存在当前浏览器中，并在页面间同步。部分高级面板和服务器返回的训练配方仍只有英文。

## 使用 Google Colab GPU

Colab 使用的是**每位用户自己的 Google 账号和算力余额**。本项目不会接收 Google 密码或 OAuth 令牌。首次使用时，在项目的 `microduck_local` 目录运行：

```bash
uv run colab usage
```

按官方 Colab CLI 的提示完成授权，确认终端显示的余额大于 0，再打开网页的 `/cloud` 页面，选择训练任务和 GPU。页面会显示账号余额、任务状态、停止按钮，训练和 ONNX 导出成功后可下载模型。停止任务和正常关闭 `duck-lab` 都会尝试释放由本项目启动的 Colab 会话。

如果余额为 0，请先在 Colab 官网核对当前账号和订阅权益。Google AI Pro 的权益和 GPU 可用性由 Google 管理；开通会员不等于任意时刻都能分配到指定 GPU。Windows 用户可直接在 Colab 打开 [训练 Notebook](notebooks/microduck_train.ipynb)。

当前版本只在训练结束后下载最终 Checkpoint 和 ONNX。训练期间自动备份到 Google Drive、断线续训以及每 30 秒更新 3D 视图尚未实现。部署到真实机器人前，应按原项目的仿真到现实流程验证导出的模型。
