# 复现依据与论文未公开细节

论文：Zhou et al., IEEE TIM 74 (2025), 3547911，DOI `10.1109/TIM.2025.3583374`。
本次重新从用户指定 PDF 提取正文并检查架构图；源文件 SHA256：
`79a39adca6baa0c6bb4da491414b31ffe4909a5cb944912425346757af06df62`。

## 方法对应

| 论文 | 本实现 |
|---|---|
| Algorithm 1 | 1024 点输入，32 点 patch，步长 16，63 个片段；信号/文本双分支 |
| 式 (1) | 每 patch 的 LSTM 最后隐藏状态，拼接 DFT 实部的线性投影，各 d/2 |
| 式 (2)–(7) | min/max/median、首尾趋势、频谱峰值/集中度、小波能量与熵 |
| 图 2 文本模板 | 无标签数值描述，含圆周自相关 top-5 lag、主频 bin、小波熵；另加入集中度 |
| 式 (8) | 正式 tokenizer + 冻结的完整预训练 BERT，mask-aware 池化 |
| 式 (9) | 投影、位置编码、跨 patch Transformer；层数/FFN 宽度可配置 |
| III-B.1 三类 embedding | 时序 patch、压缩预训练词表原型、上下文文本表示 |
| 图 1 词表压缩 | 将 V×H 词表沿 V 维线性映射到 2000×H，再映射到 d |
| 式 (10)–(13) | LayerNorm，时序 Q、文本与词表原型 K/V，多头缩放点积注意力 |
| 式 (14)–(15) | 正弦位置编码 |
| 式 (16) | 两个同宽残差 GELU 模块、平均池化、分类线性层 |
| IV-B | lr=1e-4，batch=64，最多 5 epoch，验证 loss 两轮未改善则停止 |
| 表 I / IV-D | 四种三源域组合；全训练数据 / 1% 源训练分区；未见目标域评估 |
| IV-E | time/text/multimodal 与 BERT/GPT 骨干消融 |

## 必须公开的实现选择

1. **正文和图示路线不同。** III 节说明 BERT 专用于文本语义表示；图 1 在融合后又放一个冻结 LLM。默认 `route=prose` 遵循前者；`route=figure` 增加融合→LLM 路径。两者共享原有文本编码器。图中连接关系不能唯一决定具体拓扑，不能宣称任一路线就是作者代码。
2. **式 (9) 的层数、池化和维数未说明。** 采用 token masked mean、Linear、1 层跨片段 Transformer、FFN=512；`text_context_layers=0` 可消融。旧版本省略了跨片段建模，本版本补齐这一机制。
3. **词表压缩与上下文文本如何结合未说明。** 本实现拼接成 K/V 序列；默认启用，`use_token_prototypes=false` 可消融。旧版本只在图示路径包含它，现依据 III-B.1 在正文路径也启用。
4. **未公开结构细节。** 默认 d=128、heads=8、单层 LSTM、dropout=.1、两个残差 MLP、均值池化、Adam、梯度裁剪 1.0。图中的 Flatten/BN/ReLU 与正文 GELU 不同，本实现采用式 (16) 与 IV-B 的 GELU。不存在依据充分的唯一网络宽度可用于重建全部参数。
5. **语言模型版本未说明。** BERT 采用 `google-bert/bert-base-uncased`，其 30522×768 词表与图一致。GPT 对照采用 `openai-community/gpt2`，这是可运行的自回归骨干选择，不能断言是论文使用的 GPT 版本。`text_backend=bert` 是旧接口名称，实际骨干由 `backbone` 决定。
6. **参数量不一致。** 默认正文完整路径实测 170,351,972 参数，其中冻结 BERT 为 108,891,648；压缩词表本身约 6100 万参数。论文声称约 97.6M，无法从完整 BERT+图中词表层推出；本实现不通过删减预训练层伪造匹配。
7. **patch 与窗口文本歧义。** Algorithm 1 明确逐 patch 生成文本，图 2 的 lag=1023 则来自整个 1024 点窗口。默认逐 patch；`text_scope=window` 保留窗口 token 序列，作为敏感性分析，不把二者当作相同模型。
8. **频域/小波细节。** db4、最多 2 层、symmetric 边界；短 patch 自动限层。熵为近似/细节系数各层能量概率的自然对数熵。集中度取 rFFT 最大五个频点，保留 DC。文本主频用 bin，另计算但不输入 Hz；不把不同采样率重采样。以上均为未公开细节的显式选择。
9. **文本模板。** 使用图中数值模板，不写入标签、文件名或数据集名；不加入含真实故障答案的句子。正式 tokenizer 超长即报错，避免尾部频谱/小波特征被静默截断。测试专用 tiny 路径不是预训练语义模型。
10. **原数据无法唯一还原。** 论文没有原文件列表、工况选择、截取长度、样本总量、种子、切窗外层步长和划分索引。默认外层不重叠切窗，并按独立 group 划分，组间约 80/10/10。`split_mode=window` 允许做窗口划分对照，但存在记录相关性泄漏。不能用把同一记录伪造成多个 group 的方式凑数据。
11. **噪声与类比例。** SNR=10 按 10 dB 解释，每窗使用平方均值功率和归一化高斯噪声。保留原幅值；零功率信号不加噪。划分后各分区独立下采样至四类 1:3:3:3 或三类 1:1:1。下采样时机不是作者公开细节。
12. **指标。** 给出 accuracy、macro/weighted/per-class F1、混淆矩阵，并区分全四类与真实出现类别 macro-F1。三类目标域仍保留四类输出。论文只给二分类 F1 公式，不能唯一确定其平均方式。
13. **VQA。** 根据固定 HIT/JNU/PU/CWRU 顺序计算四轴雷达原始面积 `.5×sum(adjacent products)`，归一化面积为原面积/2。论文未充分公开 VQA 的尺度，汇总值不保证与其 .868 可直接比较。
14. **基线范围。** 提供独立 WDCNN/DLinear 分类适配，使用同一数据划分/训练流程。它们不声称是作者原始配置。xLSTM/Autoformer 原论文对照配置没有复刻；未用普通 LSTM/Transformer 替代并沿用原名称。

## 软件完整与实验完成是两件事

T2MFDF 的生产路径实现真实预训练模型及各模块，无占位层或随机文本回退。自动测试、真实 BERT 梯度检查只证明工程可运行。
仍需要用户提供或下载论文涉及的真实记录、核对标签/通道/工况，才能训练并比较表 II 与图 8/9。
目前不能声称达到 ACC=93.256%、F1=.934 或任一论文迁移成绩。

原始数据源调查应分别核对 CWRU、Paderborn、JNU 官方发布信息；尤其 HIT 在论文中缺少唯一下载地址，不应用别的数据冒充。

## 接口依据

- [Transformers 4.46.3 BERT](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/bert)
- [Transformers 4.46.3 GPT-2](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/gpt2)
- [论文作者机构存档](https://pure.qub.ac.uk/en/publications/t2mfdf-an-llm-enhanced-multimodal-fault-diagnosis-framework-integ/)
