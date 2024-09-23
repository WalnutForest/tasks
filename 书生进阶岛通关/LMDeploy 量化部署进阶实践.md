# LMDeploy 量化部署进阶实践

<!-- 进度：2.1完成，接下来进行2.2. -->

## 1. 基础任务

### 1.1. 任务描述

- 使用结合W4A16量化与kv cache量化的internlm2_5-1_8b-chat模型封装本地API并与大模型进行一次对话，作业截图需包括显存占用情况与大模型回复，参考4.1 API开发(优秀学员必做)，请注意2.2.3节与4.1节应使用作业版本命令。
- 使用Function call功能让大模型完成一次简单的"加"与"乘"函数调用，作业截图需包括大模型回复的工具调用情况，参考4.2 Function call(选做)

## 2. 复现步骤

### 2.1. lmdeploy 环境准备

#### 2.1.1. 安装其他依赖包

```bash
# 安装其他依赖包
conda create -n lmdeploy  python=3.10 -y
conda activate lmdeploy
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3
```

- 结果截图

![lmdeploy环境准备](./images/task3-lmdeploy环境准备.png)

#### 2.1.2. 链接模型

```bash
# 链接模型
mkdir /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat /root/models
ln -s /root/share/new_models/OpenGVLab/InternVL2-26B /root/models
```

- 结果截图

![模型链接](./images/task3-模型链接.png)

#### 2.1.3. 启动internlm2_5-1_8b-chat

```bash
conda activate lmdeploy
lmdeploy chat /root/models/internlm2_5-1_8b-chat
```

- 输入“请你帮我生成一个以狐狸和西瓜为主角的小故事”

- 结果截图

![启动internlm2_5-1_8b-chat](./images/task3-启动internlm2_5-1_8b-chat.png)

- 显存占用情况（为21.6GB）

![显存占用情况](./images/task3-启动internlm2_5-1_8b-chat显存占用情况.png)

### 2.2. API链接测试

#### 2.2.1. 启动API服务器

```bash
conda activate lmdeploy
lmdeploy serve api_server \
    /root/models/internlm2_5-7b-chat \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```

- 终端结果截图

![启动API服务器](./images/task3-启动API服务器终端图片.png)

- 浏览器结果截图

![启动API服务器](./images/task3-启动API服务器浏览器图片.png)

#### 2.2.2. 以Gradio网页形式连接API服务器

```bash
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

- 终端结果截图

![以Gradio网页形式连接API服务器](./images/task3-以Gradio网页形式连接API服务器终端图片.png)

- 浏览器结果截图

![以Gradio网页形式连接API服务器](./images/task3-以Gradio网页形式连接API服务器浏览器图片.png)

### 2.3. 启用2.2.4 W4A16 量化+ KV cache+KV cache 量化

#### 2.3.1. 启用 W4A16 量化

```bash
lmdeploy lite auto_awq \
   /root/models/internlm2_5-1_8b-chat \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --batch-size 1 \
  --search-scale False \
  --work-dir /root/models/internlm2_5-1_8b-chat-w4a16-4bit
```

- 可以注意到对应目录`/root/models/`出现了量化后的模型`internlm2_5-1_8b-chat-w4a16-4bit`，结果截图如下：

![启用 W4A16 量化](./images/task3-启用W4A16量化.png)

