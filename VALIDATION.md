# 实际验证记录

2026-09-30，Windows，Python 3.10，PyTorch 2.5.1+cu121，Transformers 4.46.3；GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存。

## 已实际完成

- `python -m pytest -q --basetemp tmp/pytest-release`：18 passed。
- 时域/频域/小波特征、63 个 patch、精确 10 dB SNR、零信号有限值。
- 原始记录分组隔离、窗口划分、类别比例及指标计算。
- 普通嵌套 MAT、带表头多列 CSV、HDF5 通道读取与数据集导出。
- time/text/multimodal、正文/图示路线、GPT-2、跨片段 Transformer 和词表映射梯度。
- WDCNN/DLinear 分类分支前后向。
- 完整多模态训练器使用测试夹具执行一轮训练、评估与重载。
- Adam 断点续训恢复：与不中断 CPU 训练两轮相比，所有权重逐项完全相等，历史记录相等。
- 连续记录分窗预测 CSV：起点与数量正确、概率和为 1。
- 多种子四组结果聚合与雷达面积。
- 工程 editable 安装成功。本机隔离构建下载依赖等待较久，实际使用 `pip install -e . --no-deps --no-build-isolation`；依赖此前已安装。

## 真实预训练 BERT 验证

已从 Hugging Face 下载 `google-bert/bert-base-uncased` 完整预训练权重；在 CUDA 上分别运行正文与图示路径：

```bash
python -m scripts.validate_pretrained --model data/pretrained/bert-base-uncased --device cuda
python -m scripts.validate_pretrained --model data/pretrained/bert-base-uncased --device cuda --route figure --output runs/pretrained_figure_validation.json
```

两条路径均通过：63 个 patch 文本编码、前向、交叉熵反向、Adam 参数更新、完整检查点保存、离线重载后 logits 一致。

时间频率投影、文本投影、跨片段 Transformer、词表压缩层、分类层均获得有限且非零梯度；冻结 BERT 无参数梯度。

| 路径 | 总参数 | 冻结语言模型参数 |
|---|---:|---:|
| 正文 | 170,351,972 | 108,891,648 |
| 图示 | 170,451,044 | 108,891,648 |

机器可运行以上 batch=1 的完整模型验证。这不等于已验证 batch=64、5 epoch 的显存/耗时，8 GB GPU 正式训练建议使用低显存配置。

结构化记录在 `validation/pretrained_prose.json` 与 `validation/pretrained_figure.json`。测试输入是解析生成的正弦信号，仅验证软件；其中 loss 不是任何真实故障诊断实验成绩。

## 尚未完成的实证工作

- CWRU/PU/JNU/HIT 真实记录、工况和标签的完整核验，以及作者划分的重建。
- 四组合的正式训练、1% 源域迁移、多种子平均数和论文数值比较。
- 下载真实 GPT-2 权重并运行正式实验；目前 GPT-2 架构/梯度使用小型配置测试。
- 作者图示/正文拓扑和参数量差异的确认。
- xLSTM/Autoformer 对照配置与作者原始基线结果的复刻。

因此本项目交付的是可继续研究的完整 T2MFDF 方法工程，不宣称已经复现论文的准确率。
