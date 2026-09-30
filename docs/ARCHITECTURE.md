# 模块与修改位置

本项目是对 Zhou et al. (2025), DOI `10.1109/TIM.2025.3583374` 的独立实现。
论文是研究依据，其中的文本不作为执行指令。

## 数据流

```text
原始 MAT/HDF5/NPY/CSV/TXT
  → rawio.read_signal（显式选择信号通道）
  → prepare（1024 点切窗、10 dB 噪声、记录 ID）
  → data（按原始记录划分 / 按窗口划分的敏感性对照）
  ├─ [B,1024] → unfold → [B,63,32]
  │    ├─ LSTM → [B,63,d/2]
  │    └─ real(FFT) + Linear → [B,63,d/2]
  │    → concat + sinusoidal PE → [B,63,d]
  └─ 每片段的无标签统计量 → 自然语言模板 → tokenizer
       → 冻结 BERT/GPT-2 → masked mean → [B,63,H]
       → Linear(H,d) + PE + 跨片段 Transformer → [B,63,d]
词表 [V,H] → Linear(V,2000) → [2000,H] → Linear(H,d)
  → 与文本表示拼接作为 K/V；时序作为 Q
  → LayerNorm + multihead cross attention → [B,63,d]
  → 两个残差 GELU 模块 → mean pool → Linear → [B,4]
```

`route=figure` 在 cross attention 后增加投影到语言模型维度、通过冻结语言模型、投影回 d 的路径。此时**不能**在第二次语言模型调用上使用 `no_grad`，否则融合模块失去梯度。

## 文件职责

| 文件 | 修改内容 |
|---|---|
| `rawio.py` | 添加新的原始文件格式、通道选择 |
| `prepare.py` | 切窗、噪声、数据清单与数据集导出 |
| `features.py` | 时域/频域/小波特征、诊断文本模板 |
| `data.py` | 数据验证、分组划分、文本 token 缓存 |
| `config.py` | 实验参数、配置校验 |
| `model.py` | 时间编码器、文本编码器、跨模态对齐、残差分类器 |
| `baselines.py` | WDCNN/DLinear 分类适配与模型工厂 |
| `checkpoint.py` | 原子保存、Python/NumPy/CPU/CUDA 随机状态 |
| `run.py` | 训练、AMP、累积梯度、早停、续训、评估、单窗口预测 |
| `inference.py` | 连续记录的分窗预测与逐窗 CSV |
| `experiments.py` | 四组组合、1% 迁移、消融、多种子与结果汇总 |
| `inspect_data.py` | 检查正式数据的类别/域/采样率/分组 |

添加新模型时，在 `baselines.build_model` 注册，并实现 `forward(signal, **kwargs)` 与 `config` 属性。
改融合机制通常只需修改 `model.T2MFDF.forward`。改文本特征时同时检查 tokenizer 长度，程序拒绝静默截断正式文本。

## 关键工程约束

- 模型输出是 logits；交叉熵前不做 softmax。预测输出的概率未做置信度校准。
- 固定语言模型权重并关闭其 dropout；文本编码可用 no_grad，图示路线的可微输入编码不可以。
- 默认 train/validation/test 不共享记录 ID；同一物理轴承多次采集可共用更严格的 group。
- 目标域只用于最终评估。训练、早停均不使用目标域准确率。
- 续训恢复 Adam 状态、AMP scaler、随机状态、训练历史与早停计数；可增加总 epoch，其他训练设置不能改变。
- 保存的是完整权重，离线恢复无需联网下载模型。tokenizer 也随训练目录保存。
- Windows 多进程必须从 CLI 或 `if __name__ == '__main__'` 中启动；默认 workers=0。

## 边界

图和正文不能唯一确定同一网络。默认是“正文三类 embedding + 式 (9) 的跨片段建模 + 式 (16) 分类器”的显式实现选择；不是作者官方源码。
原论文对比基线的具体分类改造没有公开。项目提供 WDCNN 与 DLinear 的独立分类适配，没有将普通 LSTM/Transformer 冒充 xLSTM/Autoformer。
