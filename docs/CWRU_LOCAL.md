# CWRU 单数据集实验

最终使用目录：`D:\多模态故障\CWRU\processed\cwru16_12k_v3`。
旧的 `cwru16_fs12000_v1` 和中间检查目录 `cwru16_12k_v2` 不用于实验。

## 处理协议

16 个原始文件保留不动。正常记录 97–100 使用对应编号的 DE 通道，
按文献支持的 48 kHz 通过 scipy.signal.resample_poly 抗混叠降采样到 12 kHz；
故障记录保持 12 kHz。先重采样，再以 1024 点窗口和步长切分，最后添加噪声。
正常 413、内圈 476、外圈 475、滚动体 473，共 1837 个窗口。
标签顺序为正常 0、内圈 1、外圈 2、滚动体 3。
99.mat 只取 X099_DE_time，避免重复导入附带的 X098 通道。

依据：https://pmc.ncbi.nlm.nih.gov/articles/PMC10857163/ 的表 5。
这是文献支持的复现处理方案，不是官方逐文件采样率证明；MAT 内无明确采样率字段，
也不能声称 T2MFDF 作者使用了同样的重采样协议。此前第三方表格的 12 kHz 标注不再用作默认方案。

输出包括干净版 cwru_clean.npz、10 dB 噪声版 cwru_snr10.npz、清单、协议和检查报告。
两种版本不要合并。每条采集记录作为一个 group，不同分区不会共享同一记录。
不同负载可能共享物理轴承，不能据此声称是独立轴承测试。

重新处理时必须换一个尚不存在的输出目录：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.prepare_cwru --raw 'D:\多模态故障\CWRU' --output-dir 'D:\多模态故障\CWRU\processed\cwru16_rerun' --normal-fs 48000
```

## 第一次训练

以下命令由用户决定何时运行。准备工作没有执行真实模型训练。
打开 PowerShell，依次执行：

```powershell
cd 'C:\Users\李鑫文\Documents\ChatGPT\多模态故障诊断'
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:HF_HOME = 'D:\多模态故障\CWRU\hf_cache'
$env:HF_HUB_CACHE = 'D:\多模态故障\CWRU\hf_cache\hub'
$env:TRANSFORMERS_CACHE = 'D:\多模态故障\CWRU\hf_cache\hub'
$data = 'D:\多模态故障\CWRU\processed\cwru16_12k_v3\cwru_clean.npz'
$run = 'D:\多模态故障\CWRU\runs\single_clean_seed42'
& $py -m t2mfdf.run train --data $data --config configs/cwru_single.json --output $run --device cuda
```

首次运行需要从 Hugging Face 获取完整 BERT 权重，网络必须可达。
使用完整模型、batch=4、梯度累积16、AMP和BERT分块8；这是8GB显存适配起点，
不保证所有运行环境不会显存不足。最多5轮，验证损失选择最佳权重。
单数据集不传 --target CWRU，否则所有数据都会成为目标域，训练源域为空。
关闭论文类比例下采样，保留本子集全部窗口；这不是论文完整实验协议。
每类只有4条记录，所以按记录划分是每类2条训练、1条验证、1条测试，不能称严格80/10/10。

## 看结果

训练完成会自动用最佳权重评估测试集，结果在运行目录 metrics.json，
包括 accuracy、macro_f1 和 confusion_matrix。矩阵行是真实类别，列是预测类别。

```powershell
Get-Content "$run\metrics.json"
# 可选：重新评估同一测试集
& $py -m t2mfdf.run evaluate --data $data --checkpoint "$run\best.pt" --split test --device cuda
```

不要用 --split all 代替测试集，这会把训练样本也算进去。
中断后在同一训练命令后添加 --resume，从已保存的轮次恢复。
噪声实验改为 cwru_snr10.npz，并使用另一个运行目录。
单次结果只代表这个CWRU子集；不代表跨数据集迁移或整篇论文数值复现。
