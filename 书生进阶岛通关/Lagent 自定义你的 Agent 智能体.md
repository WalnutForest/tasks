# Lagent 自定义你的 Agent 智能体

<!-- 进度：无 -->

## 1. 基础任务

### 1.1. 任务简介

- 使用 Lagent 自定义一个智能体，并使用 Lagent Web Demo 成功部署与调用，记录复现过程并截图。

### 1.2. 复现过程

#### 1.2.1. 环境准备

- 创建环境

```bash
# 创建环境
conda create -n agent_camp3 python=3.10 -y
# 激活环境
conda activate agent_camp3
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖包
pip install termcolor==2.4.0
pip install lmdeploy==0.5.2
```

- 结果截图

![Lagent环境准备](./images/task2-Lagent环境准备.png)

- 源码下载lagent

```bash
# 创建目录以存放代码
mkdir -p /root/agent_camp3
cd /root/agent_camp3
git clone https://github.com/InternLM/lagent.git
cd lagent && git checkout 81e7ace && pip install -e . && cd ..
pip install griffe==0.48.0
```

- 结果截图

![Lagent源码下载](./images/task2-Lagent源码下载.png)

#### 1.2.2. Lagent Web Demo 使用

- 使用 LMDeploy 部署 InternLM2.5-7B-Chat，并启动一个 API Server

```bash
conda activate agent_camp3
lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --model-name internlm2_5-7b-chat
```

- 结果截图

![LMDeploy部署InternLM2.5-7B-Chat](./images/task2-LMDeploy部署InternLM2.5-7B-Chat.png)

- 在另一个窗口中启动 Lagent 的 Web Demo

```bash
cd /root/agent_camp3/lagent
conda activate agent_camp3
streamlit run examples/internlm2_agent_web_demo.py
```

- 结果截图

![Lagent Web Demo](./images/task2-LagentWebDemo.png)

- 在浏览器中打开 `http://localhost:8501/`，即可看到 Lagent Web Demo 页面

- 结果截图

![Lagent Web Demo 页面](./images/task2-LagentWebDemo页面.png)

#### 1.2.3. 基于 Lagent 自定义智能体

- 创建工具文件

```bash
cd /root/agent_camp3/lagent
touch lagent/actions/magicmaker.py
```

- 代码复制进入 `/root/agent_camp3/lagent/lagent/actions/magicmaker.py`

```python
import json
import requests

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class MagicMaker(BaseAction):
    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
    
    @tool_api
    def generate_image(self, keywords: str) -> dict:
        """Run magicmaker and get the generated image according to the keywords.

        Args:
            keywords (:class:`str`): the keywords to generate image

        Returns:
            :class:`dict`: the generated image
                * image (str): path to the generated image
        """
        try:
            response = requests.post(
                url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
                data=json.dumps({
                    "official": True,
                    "prompt": keywords,
                    "style": self.style,
                    "poseT": False,
                    "aspectRatio": self.aspect_ratio
                }),
                headers={'content-type': 'application/json'}
            )
        except Exception as exc:
            return ActionReturn(
                errmsg=f'MagicMaker exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        image_url = response.json()['data']['imgUrl']
        return {'image': image_url}
```

- 修改 `/root/agent_camp3/lagent/examples/internlm2_agent_web_demo.py` 来适配自定义工具。

```diff
from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
+ from lagent.actions.magicmaker import MagicMaker
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol

...
        action_list = [
            ArxivSearch(),
+             MagicMaker(),
        ]
```

- 启动 Web Demo

- 结果截图

![task2-Lagent文生图](./images/task2-Lagent文生图.png)

![task2-Lagent搜索论文](./images/task2-Lagent搜索论文.png)
