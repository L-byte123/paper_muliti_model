# 已下载的 16 条 CWRU 记录

通用导入器 `t2mfdf.prepare`、通道读取器 `t2mfdf.rawio` 和检查器
`t2mfdf.inspect_data` 已在原始交付中提供。本次补充的是选定文件的适配入口，
复用通用导入器，不改变模型。原始数据、处理数据和权重不提交 Git。

在项目根目录运行（输出目录必须尚不存在）：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.prepare_cwru --raw 'D:\多模态故障\CWRU' --output-dir 'D:\多模态故障\CWRU\processed\cwru16_fs12000_v1' --normal-fs 12000
```

输出 `manifest.csv`、`protocol.json`、`inspection.json`，以及
`cwru_clean.npz`（无人工噪声）和 `cwru_snr10.npz`（10 dB 高斯噪声），
各自附有导入来源 JSON。每窗 1024 点，步长 1024，随机种子 42。
两份数据是同一批记录的不同版本，不要拼接后随机划分。

选择正常 97–100、内圈 105–108、外圈 130–133、滚动体 118–121。
故障直径为 0.007 英寸，外圈位置为 6 点钟方向，包含四个负载。
标签为正常 0、内圈 1、外圈 2、滚动体 3。只选择匹配文件编号的 DE 通道，
尤其 99.mat 选择 X099_DE_time，不导入其中附带的 X098_DE_time。

## 采样率与解释限制

正常记录的 12000 Hz 是**工作假设，尚未由原始文件元数据独立确认**，依据是
[Vibdata 元数据表](https://raw.githubusercontent.com/ivarejao/vibdata/master/vibdata/raw/CWRU/CWRU.csv)。
它不是 CWRU 官方逐文件采样率证明。官方实验说明涉及 12 kHz 和 48 kHz，
不同公开处理实现对此存在差异。命令强制明确指定 `--normal-fs`，并将其写入
`protocol.json`。本入口不做重采样；正式报告频率特征相关实验前应核实该假设。
若确认正常记录为 48000 Hz，应使用新的输出目录并传入 `--normal-fs 48000`；
是否统一重采样需另行确定实验协议，不能只修改 fs 来冒充重采样。

group 是采集记录 ID。检查器验证能否按记录划分，并不会自动保存正式训练划分；
训练入口负责保存划分。不同负载可能共享同一物理轴承，因此记录互斥不代表
轴承互斥，也不能据此声称对新轴承的泛化性能。这只是 16 条记录的子集，
并非论文完整跨数据集实验。

官方文件映射：
- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
