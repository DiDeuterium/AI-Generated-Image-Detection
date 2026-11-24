# fqm · AI 生成图像检测实验

## Kaggle 0.78700 单模型

作业 7200 从随机权重训练原尺度纹理 CNN。每张图最多取 9 个去重的 128×128 图块。共享编码器处理 JPEG 重编码响应和局部残差，中心和四角负责基础分类，其余图块根据与主图块的重叠程度提供幅度受限的修正。训练时同时预测缩放再压缩前的局部亮度残差自相关，辅助损失权重为 0.10。

训练使用固定划分内的 5,760 张比赛图，另外留出 640 张校准阈值。训练 24 轮，每轮保存权重。作业 7202 在同一选模半集上比较全部 24 轮，选出第 9 轮，使用校准阈值 `0.6597230434417725` 导出单模型提交。Kaggle 得分从第 12 轮的 0.76900 提高到 0.78700。测试图仅用于最终推理。[实验记录](训练与测试分布对齐实验.md#逐轮选模与配平训练对照)

在项目根目录准备固定版本的公共代码后，在 GPU 节点的 `CV` 环境运行训练入口。

```bash
mkdir -p fqm/runs/lqy_source_b23beec fqm/logs fqm/submissions
git archive b23beec detection lqy | tar -x -C fqm/runs/lqy_source_b23beec
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
conda activate CV
python -u fqm/patch_evidence_texture.py --output fqm/runs/reproduce_anchor_context --anchor-context-only
```

权重、运行日志和提交 CSV 保存在实验机器上。训练入口是 [patch_evidence_texture.py](patch_evidence_texture.py)，逐轮选模和导出入口是 [evaluate_native_texture_base.py](evaluate_native_texture_base.py)。

## EfficientViM-M1 基线

本目录实现 [需求文档](需求文档.md) 的第一阶段：**EfficientViM-M1** 二分类基线，从随机初始化
（`weights=None` 等价，不加载任何预训练权重）开始，只使用比赛训练子集学习。训练与评测协议
遵循仓库根目录 README 的[统一实验协议](../README.md#统一实验协议)和
[ddc/实验设置.md](../ddc/实验设置.md) 的公共设置。

## 代码结构

```text
fqm/
  ssd/efficientvim.py   # EfficientViM-M1 纯 PyTorch 实现（HSM-SSD + 4 路加权分类头）
  ssd/data.py           # 划分清单读取、统一预处理、归一化统计、JPEG 重编码、DataLoader
  ssd/engine.py         # 种子、学习率调度、训练一轮、验证、指标、checkpoint
  train.py              # 训练入口：选模（原图 AUROC）→ 用最佳权重重测原图与 JPEG 版
  smoke_test.py         # 开训自检：一次前向/反向/参数更新、一次验证、checkpoint 复现
  predict.py            # 测试集推理，按 sample_submission.csv 顺序导出
  configs/efficientvim_m1.json    # 基线配置（种子 42）
  configs/norm_stats.json         # 训练子集估计的归一化参数（首次训练自动生成）
  slurm/train_efficientvim_m1.sbatch  # 作业名 test，fnlp-3090 单卡
  runs/                 # 运行产物（不入库）：log.jsonl / results.json / best.pt / last.pt
  logs/                 # Slurm 标准输出与错误
```

## 环境

- Slurm 分区 `fnlp-3090`（NVIDIA GeForce RTX 3090，24 GiB），conda 环境 `CV`。
- 实测版本：Python 3.11.17、PyTorch 2.14.1+cu126、CUDA 12.6、cuDNN 91002、Pillow 12.3.0、
  scikit-learn 1.9.1。模型不依赖 timm / mamba_ssm / Triton。
- 参数量：5.717M（论文 6.7M 是 1000 类 ImageNet 头；本任务二分类头更小，其余结构不变）。

## 与公共设置（ddc/实验设置.md）的对应关系

| 公共设置 | 本项目实现 |
| --- | --- |
| 固定划分 6400 / 1600 | `read_splits` 按 `splits/train_val.csv` 的 `split` 列读取；测试集只在 `predict.py` 出现 |
| RGB、短边 256、训练随机裁 224 + 水平翻转、验证中心裁 224 | `build_transform`（双线性 + antialias） |
| 像素 [0,1] 后归一化，统计只由训练子集按验证变换计算 | `compute_norm_stats`，结果写入 `configs/norm_stats.json` |
| JPEG 验证版：原尺寸 RGB、Pillow quality=75、subsampling=2 | `ensure_jpeg_cache`，缓存 `cache/val_jpeg_q75` |
| AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=0.01（统一） | `build_optimizer`，不做参数分组排除 |
| 有效 batch 64，microbatch 32 × 累积 2 | 配置 `micro_batch=32`、`accum_steps=2`；损失先除以累积次数 |
| 30 epoch × 100 次更新 = 3000 次，不提前停止 | `epochs`、`updates_per_epoch`、`total_updates` |
| 前 200 步 3e-6→3e-4 线性升温，其余 2800 步余弦到 3e-6 | `lr_at_step`，按优化器更新计数（跨 epoch 连续） |
| CUDA FP16 AMP + GradScaler，累积完成后更新 | `torch.autocast` + `torch.amp.GradScaler` |
| 种子 42 初筛；复跑 17/43/97 | `train.seed` + `--seed` 覆盖 |
| 选模：原图验证 AUROC，并列取较早轮次 | `select_score > best['score']`（严格大于，保留较早 epoch） |
| 报告 acc、AUROC、两类召回、混淆矩阵（行真值、列预测，顺序 0/1） | `compute_metrics`，硬分类取 logits argmax |
| 记录参数量、实际微批量、有效批量、峰值显存、每轮耗时 | `results.json` / `run_info.json` / `log.jsonl` |

## 运行命令

仓库根目录下提交（作业名固定为 `test`）：

```bash
sbatch fqm/slurm/train_efficientvim_m1.sbatch
# 复跑其它种子：参数透传给 train.py
sbatch fqm/slurm/train_efficientvim_m1.sbatch --seed 17 --run-name efficientvim_m1_seed17
```

本地等价命令（已在 `CV` 环境验证）：

```bash
python fqm/smoke_test.py --config fqm/configs/efficientvim_m1.json --batch-size 8
python fqm/train.py --config fqm/configs/efficientvim_m1.json
python fqm/predict.py --config fqm/configs/efficientvim_m1.json \
    --checkpoint fqm/runs/efficientvim_m1_seed42/best.pt
```

调试时可加 `--epochs 1 --updates-per-epoch 1 --limit-train 128 --limit-val 64`，
这些参数只在调试配置中使用，正式结果不加。

## 产物位置

- 逐轮日志：`runs/<run_name>/log.jsonl`（含每轮训练损失/准确率/学习率/耗时/峰值显存，
  以及验证损失、准确率、AUROC、两类召回率、混淆矩阵）
- 汇总：`runs/<run_name>/results.json`（参数量、批量、总更新数、耗时、显存、最佳 epoch、
  原图与 JPEG 版指标）
- checkpoint：`runs/<run_name>/best.pt`（最佳原图 AUROC）、`runs/<run_name>/last.pt`
- 逐图概率：`runs/<run_name>/val_predictions_{orig,jpeg}.csv`（image_id、label、prob_fake）

## 结果

基线（EfficientViM-M1，种子 42，最佳 epoch 21）：

| 验证版本 | 准确率 | AUROC | 真实类召回 | 生成类召回 | 混淆矩阵 [[TN, FP], [FN, TP]] |
| --- | --- | --- | --- | --- | --- |
| 原图 | 0.6981 | 0.7802 | 0.7488 | 0.6475 | [[599, 201], [282, 518]] |
| 统一 JPEG 重编码（q=75，4:2:0） | 0.7013 | 0.7786 | 0.7550 | 0.6475 | [[604, 196], [282, 518]] |

调优后（**CAP 伪迹先验模块** + 原尺度裁剪 320 + 60 epoch + EMA，4 个种子均值 ± 标准差，
见 [最终结果](最终结果.md)）：

| 验证版本 | 准确率 | AUROC | 真实类召回 | 生成类召回 |
| --- | --- | --- | --- | --- |
| 原图 | **0.9394 ± 0.0036** | **0.9851 ± 0.0010** | 0.9547 ± 0.0139 | 0.9241 ± 0.0143 |
| 原图 + 五裁剪/翻转 TTA | **0.9598 ± 0.0034** | **0.9935 ± 0.0003** | — | — |
| **4 种子 + TTA 集成** | **0.9663** | **0.9961** | 0.9750 | 0.9575 |
| 统一 JPEG 重编码（q=75，4:2:0） | 0.7711 ± 0.0096 | 0.8816 ± 0.0137 | — | — |
| 重编码 + TTA 集成 | 0.7669 | 0.9373 | 0.9925 | 0.5413 |

单个种子（种子 42）混淆矩阵（行真值、列预测，顺序 0/1）：
原图 `[[766, 34], [56, 744]]`、JPEG 版见 `runs/final/final_w5_cap60_seed42/results.json`；
集成后原图 `[[780, 20], [34, 766]]`。

**重要限制**：先验模块把数据集「真实=JPEG、生成=PNG」的捷径显式化了，域内精度大幅提升，
但统一重编码验证集会退化。训练时加 p=0.2 的 JPEG 重编码增强（`w6_cap60_jpegaug`）可以把
重编码准确率从 0.7756 拉回 **0.9150**，域内只掉 2.1 个点（0.9231）——见
[最终结果](最终结果.md) 的「缓解方案」一节。

设备与开销：RTX 3090 单卡。参数量 5.719M（含模块 16 个额外参数），微批量 32 × 2 = 有效批量 64，
320 输入下峰值已分配显存 1.9 GB、每轮 20.4 s、60 epoch 约 31 分钟（4 个种子各一次）；
基线（224 输入、30 epoch）峰值 415 MB、每轮 13.6 s、共 9.37 分钟。

测试集提交：
`runs/final/submission_ensemble_tta.csv`（4 种子 + TTA 集成，1000 行，已校验）、
`runs/final/final_w5_cap60_seed42/submission.csv`（单模型）、
`runs/tune/w6_cap60_jpegaug/submission.csv`（JPEG 增强版，鲁棒性优先）。

逐轮日志与复跑命令见 [实验记录](实验记录.md)（基线）、[调优记录](调优记录.md)
（六轮共 40 组配置 + 4 种子复跑 + TTA/集成）；模块设计见 [模块说明](模块说明.md)；
最优 checkpoint 位于 `runs/final/final_w5_cap60_seed17/best.pt`（AUROC 最高）。

## CAP 先验模块（创新点）

`fqm/ssd/prior.py` 实现了**压缩/合成伪迹先验**（Compression & synthesis Artifact Prior）：
用固定算子显式估计「8×8 网格相位搜索的块效应」与「高频残差能量」，作为 8 个先验通道与 RGB
拼接后送入 EfficientViM 骨干；模块只有 16 个可学参数，不改变骨干结构。设计与消融见
[模块说明](模块说明.md)，开关方式是在配置里加

```json
"model": {"name": "efficientvim_m1", "num_classes": 2,
          "prior": {"parts": ["block", "highfreq"]}}
```

实验配置为 `fqm/tune/configs/w5_*.json`（完整模块、仅高频、仅块效应、梯度截断、
错误块大小 7 的对照、与标签平滑/更大模型的交互）。


```text
fqm/tune/gen.py             # 生成四轮调参配置（manifest_w1..w4.txt）
fqm/tune/summarize.py       # 汇总各 run 的验证指标
fqm/tune/make_final.py      # 按 AUROC 选最优配置，生成多种子复跑配置
fqm/tune/aggregate_final.py # 多种子均值 ± 标准差 + TTA 结果
fqm/tune/pipeline.sh        # 后台流水线：等队列 → 选优 → 复跑 → 汇总
fqm/slurm/tune_efficientvim.sbatch   # 调参阵列作业（1 GPU/任务）
fqm/slurm/final_efficientvim.sbatch  # 最终复跑（训练 + TTA）
fqm/slurm/predict_best.sbatch        # 测试集推理与提交文件校验
fqm/evaluate_tta.py         # 五裁剪 + 翻转的多视图 TTA 评测
```
