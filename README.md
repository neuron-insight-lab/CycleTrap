## 🚀 简介

这是我们论文《循环诱导驱动的多模态大语言模型可用性压力测试方法》的代码库。

## 📂 目录结构

```
.
├── assets/
│   └── overview.png
├── baseline/
│   ├── LingoLoop.py
│   ├── README.md
│   ├── VerboseVideo.py
│   └── VerboseImage.py
├── utils/
│   ├── __init__.py
│   ├── opt_utils.py
│   └── vision_utils.py
├── README.md
├── requirements.txt
└── main.py
```

## 🎓 摘要

多模态大语言模型凭借其强大的跨模态理解能力，已在图像描述、视频理解等任务中广泛应用，然而其高昂的推理计算开销也使得相关服务面临以资源耗尽为目标的可用性安全威胁。对此类威胁进行有效的压力测试，是评估和增强系统可用性安全性的重要手段。现有的压力测试方法普遍采用推迟生成结束词元以延长输出序列的策略，但存在测试强度有限与优化效率较低的问题。为此，本文提出一种面向多模态大语言模型的高效压力测试框架CycleTrap。该框架通过显式改变模型的输出结构，诱导模型进入自我强化的循环生成状态，使其持续输出重复内容直至达到最大生成长度，从而显著放大模型推理能耗与响应延迟，以此充分模拟极端推理负载场景。具体而言，本文首先设计**上下文感知的种子提取机制**，依据模型原始输出动态选取初始重复种子，以提高循环诱导成功率；随后进一步提出**循环诱导机制**，利用自回归生成过程中历史输出对当前预测的累积效应，逐步增强重复种子的生成概率，最终形成稳定的循环输出模式。大量实验结果表明，CycleTrap在多个图像与视频数据集及多种主流模型上均表现出显著优势，其诱导生成的输出长度最高可达原始输入的20倍，同时对抗性测试样本的优化开销较基线方法降低约7倍。

![overview](https://github.com/neuron-insight-lab/CycleTrap/raw/main/assets/overview.png)


## 🛠️ 安装

### 🔧 环境准备

```bash
conda create -n CycleTrap python=3.12 -y
conda activate CycleTrap
pip install -r requirements.txt
```

### 🔨 模型准备

您可以从 [huggingface](https://huggingface.co/) 下载所需的 MLLM，例如 [Qwen2.5-VL-7B-Instrcut](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)、[InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b) ([-Video](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/instructblipvideo)) 和 [LLaVA-NeXT](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)（[-Video](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)），并在 [utils/vision_utils.py](https://github.com/neuron-insight-lab/CycleTrap/blob/main/utils/vision_utils.py#L19) 文件的相应位置填写您的模型路径。

### 📊 模型准备

我们选择了两个图像模态数据集：[MS-COCO](https://cocodataset.org/#download)、[ImageNet](https://image-net.org/download-images.php)；以及两个视频模态数据集：[TGIF](https://github.com/raingo/TGIF-Release) 和 [MSVD](https://github.com/jpthu17/EMCL)。分别选择了 200 张图像或 100 个视频作为评估数据集。下载后，请将数据集路径填写到 [utils/opt_utils.py](https://github.com/neuron-insight-lab/CycleTrap/blob/main/utils/opt_utils.py#L21) 文件中的相应位置。


## 💡 快速开始

您可以运行以下命令来创建由 CycleTrap 生成的对抗性视觉输入，以诱导MLLM进行重复生成。（以 Qwen2.5-VL-7B-Instruct 模型 和 COCO 数据集为例）

```bash
python main.py \
      --model_name "Qwen2.5-VL-7B" \
      --data_name "coco" \
      --data_len 200    \
      --segment_len 5 \
      --max_new_tokens 1024 \
      --steps 300 \
      --root_dir "save/CycleTrap"
```

你可以运行“``python main.py -h`` 命令来了解每个参数的含义。