# T2MFDF 多模态故障诊断：可修改的研究工程

对应 Zhou et al., **T2MFDF: An LLM-Enhanced Multimodal Fault Diagnosis Framework Integrating Time-Series and Textual Data**, IEEE TIM 2025，[DOI](https://doi.org/10.1109/TIM.2025.3583374)。

本项目实现原始数据读取、信号与文本特征、预训练语言模型、跨模态对齐、残差分类、训练/续训、评估、连续记录预测、四组跨域实验和消融。正式入口使用真实数据与完整预训练 BERT。小型随机语言模型只用于自动测试。

**这是独立方法复现，不是作者官方源码。尚未完成四个真实数据集的论文数值复现。** 作者未给出完整数据文件清单、划分种子和若干网络参数；实现选择见 [REPRODUCTION.md](REPRODUCTION.md)，实际检查见 [VALIDATION.md](VALIDATION.md)。

## 1. 安装

Python 3.10/3.11，先安装适合本机 CUDA 的 PyTorch 2.3–2.5，再执行：

```bash
python -m pip install -e ".[test]"
```

依赖锁定 Transformers 4.46.3，与当前验证环境一致。CPU 可运行，正式训练推荐 GPU。Windows 未激活虚拟环境时，把下文 `python` 替换为 `.venv\Scripts\python.exe`。

## 2. 准备真实数据

统一标签 `0=正常、1=内圈、2=外圈、3=滚动体`。PU/HIT 使用前三类。原始数据须由使用者从数据发布机构获取；论文未给出足以唯一识别 HIT 版本的下载信息。

建立 CSV 清单，每行是一条原始采集记录：

| 字段 | 含义 |
|---|---|
| path | 原文件路径；相对路径以清单所在目录为基准 |
| key | MAT/NPZ/HDF5 变量或路径；嵌套结构示例 `record/Y/6/Data`，数组下标从 0 开始 |
| label | 经核对的标签，不从文件名猜测 |
| domain | CWRU、PU、JNU、HIT |
| fs | 实际采样率，论文分别为 12000、64000、50000、25000 Hz |
| group | 物理记录或轴承 ID；同一记录的多份导出必须使用相同 ID |
| column | 可选，多列矩阵中的通道序号，从 0 开始 |
| skiprows | 可选，CSV/TXT 要跳过的表头行数 |
| delimiter | 可选，文本分隔符；CSV 默认逗号 |

支持普通 MAT、MAT v7.3/HDF5 引用结构、NPY/NPZ、CSV/TXT/DAT。路径与通道示例见 [examples/manifest_nested.csv](examples/manifest_nested.csv)，示例标签须根据你实际记录核对。每个域×类别默认需要至少三个独立 group。

```bash
python -m t2mfdf.prepare --manifest examples/my_manifest.csv --output data/bearings.npz --snr 10 --stride 1024
python -m t2mfdf.inspect_data --data data/bearings.npz
```

每窗 1024 点，默认加入 10 dB 高斯噪声。`--clean` 不加噪声。保留真实采样率与幅值，不做跨数据集重采样或测试集归一化。现有输出不会被覆盖。

## 3. 正式训练与续训

```bash
python -m t2mfdf.run train --data data/bearings.npz --config configs/paper.json --target HIT --output runs/main --device cuda
# 中断后恢复，或明确增加总 epoch；其他配置必须保持一致
python -m t2mfdf.run train --data data/bearings.npz --config configs/paper.json --target HIT --output runs/main --device cuda --resume
```

首次运行下载 `google-bert/bert-base-uncased`。离线训练可把配置中的 `bert_model` 改为本地模型目录。默认完整 BERT、63 个 patch、词表压缩到 2000 个原型、跨片段 Transformer、跨模态注意力和残差分类器。训练参数采用论文的 lr=1e-4、batch=64、最多 5 epoch、patience=2。

8 GB 显存可使用 `configs/low_memory.json`：每批 4 窗、累积 16 次达到有效 batch=64，BERT 分块 8、启用 AMP。它保留完整语言模型与网络结构；浮点精度和分批会带来数值差异。token 缓存会消耗 CPU 内存，可用 `cache_tokens=false` 关闭。

每次训练保存 `config.json`、`splits.npz`、`history.json`、`best.pt`、`last.pt`、`metrics.json` 和 `tokenizer/`。`last.pt` 包括优化器、随机状态和早停状态；`best.pt` 用于最终评估。检查点较大，不应提交 Git。

## 4. 四组实验、多种子和消融

```bash
# 四组组合 × 全量/1% 源域训练 × 三个随机种子
python -m t2mfdf.experiments --data data/bearings.npz --config configs/paper.json --output runs/paper --seeds 42 43 44 --fractions 1 0.01
# BERT/GPT、模态、结构以及两种独立分类基线
python -m t2mfdf.experiments --data data/bearings.npz --output runs/ablations --variants full time text gpt2 figure no-context no-prototypes wdcnn dlinear --seeds 42 43 44 --fractions 1 0.01
# 中断续跑：加 --resume；只重新汇总：
python -m t2mfdf.experiments --output runs/paper --summarize-only
```

| 组合 | 源域 | 留出目标域 |
|---|---|---|
| Dataset 1 | CWRU + PU + JNU | HIT |
| Dataset 2 | CWRU + PU + HIT | JNU |
| Dataset 3 | CWRU + JNU + HIT | PU |
| Dataset 4 | PU + JNU + HIT | CWRU |

默认按 group 划分源域 80/10/10，组大小不同会导致窗口比例近似而非严格相等。各分区独立按四类 1:3:3:3、三类 1:1:1 下采样。1% 从源训练分区抽取，目标数据不参与优化或早停。

`configs/paper_window_split.json` 提供窗口随机划分敏感性对照，同一记录可跨分区，存在相关性泄漏，不能与严格分组结果混为一谈。论文未明确其划分单位。

输出 `results.csv` 与 `summary.json`，按实验/分区汇总 ACC、macro-F1、仅真实出现类别 macro-F1、均值和样本标准差。目标域仍保留四类输出，不用目标标签屏蔽类别。另计算四轴 transfer-F1 雷达图面积及归一化面积，论文 VQA 的归一化定义不充分，不能直接声称数值等价。

## 5. 评估与真实记录预测

```bash
python -m t2mfdf.run evaluate --data data/bearings.npz --checkpoint runs/main/best.pt --split test --device cuda
python -m t2mfdf.run evaluate --data data/external.npz --checkpoint runs/main/best.pt --split all --device cuda
python -m t2mfdf.run predict --data data/window.npy --fs 12000 --checkpoint runs/main/best.pt
python -m t2mfdf.inference --input data/continuous.mat --key X097_DE_time --fs 12000 --stride 1024 --checkpoint runs/main/best.pt --output runs/predictions.csv --device cuda
```

预测不重复加噪；连续输入尾部不足 1024 点会丢弃。CSV 给出起止采样点、时间、预测类别与各类概率。评估旧划分时验证数据 SHA256，避免误用其他数据的索引。

## 6. 修改与验证

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 给出完整数据流与修改位置；[REPRODUCTION.md](REPRODUCTION.md) 逐项对应论文和实现假设。

```bash
python -m pytest -q --basetemp tmp/pytest
python -m scripts.validate_pretrained --model google-bert/bert-base-uncased --device cuda
```

第二条使用真实 BERT 权重检查完整网络梯度，输入为测试信号，因此不测量诊断准确率。测试通过不能替代真实数据实验。

## 研究范围

T2MFDF 的各核心模块、训练和推理均为可执行实现，没有以 TODO 或随机特征代替正式分支。BERT/GPT-2、模态和结构消融可以独立运行。WDCNN/DLinear 是独立分类适配；论文未公开其对比网络配置，本项目没有完整复刻 xLSTM/Autoformer 对照，因此不能称为“论文所有表格的数值复现”。没有包含作者论文 PDF、原始数据或预训练权重，也没有把合成测试的成绩当作研究结论。
