# OpenCompass 评测 InternLM-1.8B 实践

<!-- 进度：无 -->

## 1. 基础任务

### 1.1. 任务概述

- 使用 OpenCompass 评测 internlm2-chat-1.8b 模型在 ceval 数据集上的性能，记录复现过程并截图。

### 1.2. 复现过程

#### 1.2.1. 环境准备

- 创建环境

```bash
# 创建虚拟环境
conda create -n opencompass python=3.10
conda activate opencompass
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装 OpenCompass
cd /root
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
apt-get update
apt-get install cmake
pip install -r requirements.txt
pip install protobuf
```

- 下载数据集

```bash
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```

#### 1.2.2. 使用命令行配置参数法进行评测

打开`opencompass`文件夹下`configs/models/hf_internlm/的hf_internlm2_chat_1_8b.py`,贴入以下代码

```python
from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-1.8b-hf',
        path="/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",
        tokenizer_path='/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=100,
        min_out_len=1,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```

- 启动测评

```bash
#环境变量配置
export MKL_SERVICE_FORCE_INTEL=1
#或
export MKL_THREADING_LAYER=GNU

# 运行评测
python run.py --datasets ceval_gen --models hf_internlm2_chat_1_8b --debug
```

- 查看评测结果

![OpenCompass评测结果](./images/OpenCompass命令行配置参数法评测结果.png)

#### 1.2.3. 使用配置文件修改参数法进行评测

- 创建配置文件

```bash
# 创建配置文件
cd /root/opencompass/configs
touch eval_tutorial_demo.py
```

- py文件内容如下：

```python
# eval_tutorial_demo.py
from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .models.hf_internlm.hf_internlm2_chat_1_8b import models as hf_internlm2_chat_1_8b_models

datasets = ceval_datasets
models = hf_internlm2_chat_1_8b_models
```

- 启动测评

```bash
# 运行评测
cd /root/opencompass
python run.py configs/eval_tutorial_demo.py --debug
```

- 查看评测结果

![OpenCompass评测结果](./images/OpenCompass配置文件修改参数法评测结果.png)

## 2. 进阶任务（未完成）

### 2.1. 任务概述

- 使用 OpenCompass 进行主观评测（选做）
- 使用 OpenCompass 评测 InternLM2-Chat-1.8B 模型使用 LMDeploy部署后在 ceval 数据集上的性能（选做）
- 使用 OpenCompass 进行调用API评测（优秀学员必做）
