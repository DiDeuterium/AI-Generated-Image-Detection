# 成员 A · 经典 CNN 训练与复现

本路线从随机权重训练 ResNet-18 和 MobileNetV2，使用 6,400 张训练图和 1,600 张验证图。配置默认沿用小组的 224 × 224 输入、有效批量 64、AdamW、200 步预热与余弦退火。两种骨干各自训练 30 轮，初筛种子为 42。 [成员分工](需求文档.md) · [统一实验设置](../ddc/实验设置.md) · [实验配置](config.json)

## 数据与环境

从比赛发布的数据仓库下载 `dataset.zip`、`train_labels.csv` 和 `sample_submission.csv`。把图像解压为 `dataset/dataset/train/` 和 `dataset/dataset/test/`，两个 CSV 放在 `ai-generated-image-detection-benchmark/`。固定划分随代码提供。 [正式图像与模板](https://huggingface.co/datasets/emiliiia/ai-generated-image-detection-benchmark/tree/main) · [小组数据说明](../doc/项目与数据说明.md)

从仓库根目录运行。与小组环境对齐可使用 Python 3.12、PyTorch 2.6.0 和 torchvision 0.21.0。RTX 4060 的 CUDA 12.4 环境可按下列命令安装。CPU 环境把 PyTorch 源的 `cu124` 换成 `cpu`。 [PyTorch 版本与安装命令](https://pytorch.org/get-started/previous-versions/#v260)

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r lqy/requirements.txt
```

## 正式实验

```sh
python -m lqy.cnn --device cpu prepare
python -m lqy.cnn --device cuda suite --output lqy/runs/formal --resume
```

`prepare` 核对固定划分的类别数及文件路径，再从训练集的中心裁剪像素计算 RGB 均值和总体标准差，写入 `lqy/runs/normalization.json`。各模型和两套验证共用这一份统计。正式数据放在其他目录时，可通过 `--image-root`、`--manifest` 和 `--norm` 指定路径。

`suite` 先跑 ResNet-18、再跑 MobileNetV2，以各自最佳权重的原图 AUROC 选胜出配置。同分依次比较原图准确率和参数量，参数较少者优先。用种子 17、43、97 复跑胜出配置，并用种子 42 单独训练一组压缩对照。该对照在原尺寸 RGB 图像上，对两类都以 50% 的概率施加 Q75、4:2:0 JPEG 重编码。训练图几何增强为随机裁剪和水平翻转。

在同一轮实验中保持两种模型的微批量一致。若设备需要改用 16 × 4，可在 `suite` 或 `train` 命令加入 `--microbatch 16`。CUDA 和 Intel XPU 使用 FP16 自动混合精度。可以显式指定 `--device cpu`，或在适配的 PyTorch 环境指定 `--device xpu`，实际硬件和精度记录在结果中。

```sh
python -m lqy.cnn --device cuda train --model resnet18 --seed 42 --output lqy/runs/resnet18-s42 --epoch-limit 1
python -m lqy.cnn --device cuda train --model resnet18 --seed 42 --output lqy/runs/resnet18-s42 --resume
```

第一条命令用正式预算运行第一轮，可用来检查显存和速度。第二条命令使用相同设置继续训练。每轮保存在 `last.pt`，意外中断后从最近完整的一轮恢复。日志记录实际参数更新和 FP16 溢出跳过的次数。

## 验证与交付

- `comparison.md` 汇总两套模型和三个重跑种子的验证表现、参数量、微批量、每轮训练时间和设备显存。三个种子报告均值和样本标准差。重编码结果衡量模型在同一图像上对再次 JPEG 压缩的敏感程度。
- 每个运行目录有 `config.json`、`epochs.csv`、`summary.json`、`best.pt` 和 `val_predictions.csv`。每轮原图 AUROC 最高的权重胜出，同分取较早轮次。完成训练后加载 `best.pt` 评估同一批验证图的原图和 Q75 JPEG 版本，以图像 ID 对齐分数和类别。
- 提交组内报告时，同步逐轮日志和结果表，并将 `best.pt` 放在组员可访问的位置。附上训练设备与精度、归一化 JSON 和代码提交号。不同设备的耗时需按设备比较。

下面以 ResNet-18 胜出为例。用于组内最终推理的 CNN checkpoint 地址在 `selection.json`，测试集按比赛模板逐行输出离散类别。

```sh
python -m lqy.cnn --device cuda evaluate --checkpoint lqy/runs/formal/resnet18-s42/best.pt --output lqy/runs/reevaluation
python -m lqy.cnn --device cuda export --checkpoint lqy/runs/formal/resnet18-s42/best.pt --output lqy/runs/submission.csv
```

其他骨干可复用 `detection.data` 的清单、变换、归一化和数据加载器，以及 `detection.training.fit` 的训练与评估入口。模型需返回形状为 `[batch, 2]` 的 logits。

## 本次实验记录

正式实验使用 Intel Arc XPU、PyTorch 2.11.0+xpu 和 torchvision 0.26.0+xpu，训练源码提交为 `b6b9505`。在同一环境下重跑时，可用 `python -m lqy.cnn --device xpu --workers 4 --threads 4 suite --norm lqy/results/normalization.json --output lqy/runs/reproduced --resume`。 [实验结果与逐轮日志](results/README.md)

三个独立种子的 ResNet-18 权重也可以按 AI 类概率做等权集成。以下命令先根据三组已有的配对验证预测计算集成指标，随后按模板导出测试预测。

```sh
python -m lqy.ensemble --device xpu --workers 4 --runs lqy/runs/formal/resnet18-s17 lqy/runs/formal/resnet18-s43 lqy/runs/formal/resnet18-s97 --output lqy/runs/formal/ensemble-three-seeds --export
```

## 原尺度裁剪试验

扩展配置以 ResNet-18 为骨干，随机初始化，使用相同的 6,400/1,600 划分。先解码 RGB，再在原尺度上裁剪 320 × 320 图块。短边不足 320 时反射填充。把图块中每个 2 × 2 像素单元分解为 RGB 均值及三个方向残差，送入 ResNet-18 的 12 通道第一层。配置包含 50% 概率的 Q75 JPEG 增强和衰减系数为 0.99 的 EMA，训练预算上限是 60 轮。 [试验配置](native320.json)

第一条命令试跑种子 42 的前 12 轮并保留断点。第二条命令用五裁剪与翻转评估阶段最佳权重，也会在同一轮次比较 EMA 权重和普通权重。

```sh
python -m lqy.native --device xpu train --seeds 42 --epoch-limit 12 --output lqy/runs/native320
python -m lqy.native --device xpu preview --run lqy/runs/native320/resnet18-native-s42 --compare-ema --output lqy/runs/native320/preview-12
```

对已有统一基线权重评估五裁剪与翻转时，使用 `python -m lqy.tta`。各模型将同一图像十个视图的 AI 类概率取均值。验证集原图和 JPEG Q75 按 ID 对齐。

四个种子的正式优化结果已完成。每个种子使用 60 轮训练预算，权重按各自的原图验证 AUROC 选定。以下 PowerShell 命令在新目录复现四种子训练、配对 TTA 评估和最终测试集成。 [四种子成绩、逐轮日志与权重校验值](results/optimized/README.md)

```powershell
$nativeOutput = 'lqy/runs/native320-reproduced'
python -m lqy.native --device xpu train --output $nativeOutput
$nativeRuns = @(17, 42, 43, 97 | ForEach-Object { "$nativeOutput/resnet18-native-s$_" })
python -m lqy.native --device xpu evaluate --runs @nativeRuns --output "$nativeOutput/final-tta"
python -m lqy.native --device xpu export --runs @nativeRuns --output "$nativeOutput/submission.csv"
```

## 从零训练的纹理 ResNet-18

`lqy.robust` 从随机权重训练。ResNet-18 在两层卷积特征上计算通道相关性，并将纹理统计和末层特征一起用于分类。训练继续使用固定划分的 6,400 张图。缩放、纵横比、模糊和 JPEG 质量采用与类别无关的随机分布。压力验证固定使用 4:3 和 3:4 两种比例，尺寸设为 32 像素整数倍，JPEG 质量设为 70 和 90。 [纹理特征的设计参考](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Global_Texture_Enhancement_for_Fake_Face_Detection_in_the_Wild_CVPR_2020_paper.html)

固定验证集每类按哈希顺序各分出 400 张选模和 400 张留出样本。按选模集增强图准确率选取权重，同分时比较 AUROC。留出对照在同一批图像上使用十视图 TTA。训练和验证从带标签目录读取图像，测试目录用于最终推理导出。

```powershell
$robustRun = 'lqy/runs/robust-course/texture320-s17'
$reference = @(17, 42, 43, 97 | ForEach-Object { "lqy/runs/native320/resnet18-native-s$_" })
python -m lqy.robust --device xpu train --epochs 18 --seed 17 --output $robustRun
python -m lqy.robust --device xpu --workers 0 audit --candidate $robustRun --reference @reference --output lqy/runs/robust-course/holdout
python -m lqy.robust --device xpu --workers 0 export --runs $robustRun --output lqy/runs/robust-course/submission.csv
```

早期缩放与压缩增强试验的 Kaggle 准确率为 54.1%，旧集成为 54.8%。该试验的压力尺寸参考了测试图的观察结果，留作诊断记录。 [早期压力验证指标](runs/robust/holdout/metrics.json)

纹理版训练 18 轮，选中第 18 轮。Kaggle 准确率为 54.0%。下表使用留出的 800 张验证图，并对两个模型使用相同的十视图推理。

| 验证条件 | 旧四模型集成 | 纹理单模型 |
| --- | ---: | ---: |
| 原图 | 94.25% | 80.25% |
| 448 × 336，JPEG Q70 | 77.38% | 76.25% |
| 384 × 512，JPEG Q90 | 73.12% | 77.00% |

[逐轮日志](runs/robust-course/texture320-s17/epochs.csv) · [留出集指标](runs/robust-course/holdout/metrics.json) · [Kaggle 提交 CSV](runs/robust-course/submission.csv)

轻量 CNN 进一步用标注数据做尺寸整组留出。每次从训练中排除一组 AI 图，验证时用该组 AI 图和固定的同数量真实图。各方法都统一重编码为 JPEG Q75。下表的尺寸按解码后的原图分组，只作图像来源的代理。

| 训练中留出的 AI 图尺寸 | 验证集大小 | 整图裁缩到 128 | 原尺度中心裁剪 128 | 原尺度五裁剪 128 |
| --- | ---: | ---: | ---: | ---: |
| 1024 × 1024 | 212 | 52.83% | 71.23% | 61.32% |
| 512 × 512 | 602 | 47.51% | 63.79% | 64.95% |

五裁剪方案在训练时随机取一个视图，验证时平均五个视图的预测。 [尺寸留出实验脚本](runs/robust-course/cohort_probe.py)

## 原尺度 128 像素小块

RGB 与残差小块的两种子短训版在 Kaggle 得到 56.3%。随后对比尺寸组均衡采样，以及在同样采样设置上约束 JPEG Q75 和 Q90 的预测一致性。两个方法均从随机权重训练 6 轮。尺寸分组用于训练采样与验证留出。 [试分配置](runs/night-train/2026-11-21/online-probe/summary.json) · [数据偏差研究](https://arxiv.org/abs/2403.17608)

| 留出组 | 原训练方式 | 均衡采样 | 均衡采样 + 一致性 |
| --- | ---: | ---: | ---: |
| 512 × 512 | 72.3% | 66.9% | 57.8% |
| 1024 × 1024 | 65.6% | 66.5% | 65.6% |
| 256 × 256，复核 | 72.9% | 65.0% | |

表中对两个种子和两种 JPEG 质量的 AI 类概率取均值。按 512 和 1024 两组中的最低准确率选出均衡采样，在完整的 6,400 张训练图上重训并导出线上对照。256 组的复核显示该改动的收益不稳定。 [实验指标](runs/group-patch/round-01/summary.json)

```powershell
python -m lqy.group_patch --device xpu --output lqy/runs/group-patch/reproduced
```

### 频谱输入对照

保留原尺度五裁剪、JPEG Q75、空间残差、CNN 结构和训练配置，只把 RGB 通道换成频谱。频谱由加窗图块计算，并屏蔽 8 像素块周期对应的频点。两方案在每个留出组上使用相同的图像，组间的真实验证图互不重叠。

| 留出组 | 原方案准确率 | 频谱准确率 | 原方案 AUROC | 频谱 AUROC |
| --- | ---: | ---: | ---: | ---: |
| 512 × 512 | 70.76% | 66.61% | 0.7690 | 0.7201 |
| 1024 × 1024 | 65.09% | 69.34% | 0.7401 | 0.7576 |
| 256 × 256 | 70.37% | 69.19% | 0.7691 | 0.7511 |

两项指标同时提升的条件只在 1024 组满足，试验按预定门槛在筛选阶段结束。配对准确率差异和重采样区间见结果文件。 [频谱对照结果](runs/spectrum-patch/round-01/summary.json)

```powershell
python -m lqy.spectrum_patch --device xpu --output lqy/runs/spectrum-patch/reproduced
```

### JPEG 压缩历史匹配

空间与频谱等权集成在 Kaggle 得到 56.5%。本组另做压缩历史对照，固定空间模型、原尺度五裁剪、训练轮数和 JPEG Q75 推理。仅在训练时用同一训练子集内真实图的量化表和色度采样给 AI 图补一次 JPEG 编码，随后按原流程裁剪。验证和测试统一走原来的 Q75 流程。该试验用于检查两类图像的二次压缩差异是否影响泛化。 [JPEG 偏差研究](https://arxiv.org/html/2403.17608v2#S4.SS1)

| 留出组 | 原方案准确率 | 匹配后准确率 | 原方案 AUROC | 匹配后 AUROC |
| --- | ---: | ---: | ---: | ---: |
| 512 × 512 | 70.76% | 71.26% | 0.7690 | 0.7702 |
| 1024 × 1024 | 65.09% | 63.68% | 0.7401 | 0.7661 |
| 256 × 256 | 70.37% | 67.00% | 0.7691 | 0.7299 |

压缩历史匹配版的 Kaggle 准确率为 54.8%。 [压缩历史对照结果](runs/jpeg-history/round-01/summary.json)

```powershell
python -m lqy.jpeg_history --device xpu --output lqy/runs/jpeg-history/reproduced
```

### 局部残差共现

从原尺度五裁剪提取亮度、色差的一阶与二阶残差。对量化后的相邻三点统计联合分布，跳过横跨 8 像素 JPEG 块边界的三点。CNN 从随机权重学习这些统计，每个种子训练 6 轮。 [残差共现方法参考](https://dde.binghamton.edu/kodovsky/pdf/TIFS2012-SRM.pdf)

| 留出组 | 原方案准确率 | 共现版准确率 | 原方案 AUROC | 共现版 AUROC |
| --- | ---: | ---: | ---: | ---: |
| 512 × 512 | 70.76% | 67.44% | 0.7690 | 0.8119 |
| 1024 × 1024 | 65.09% | 69.34% | 0.7401 | 0.8407 |
| 256 × 256 | 70.37% | 67.68% | 0.7691 | 0.7475 |

残差共现校准版的 Kaggle 准确率为 50.2%。 [残差共现结果](runs/residual-cooc/round-01/summary.json) · [统一阈值复核](runs/residual-cooc/round-01/calibrated/summary.json)

```powershell
python -m lqy.residual_cooccurrence --device xpu --output lqy/runs/residual-cooc/reproduced
```

### 全图与原尺度图块联合训练

全图分支使用随机初始化的 ResNet-18，局部分支保留五个 128 像素原尺度图块。两路特征融合后分类，两路各有辅助分类损失。训练采用统一水平翻转、两轮学习率预热和余弦衰减。每轮保存模型、优化器状态和日志。

先在 512 和 1024 整组留出上各训练 12 轮，与线上 56.5% 的空间和频谱等权方案在同一批标注图上对照。两组准确率和 AUROC 都需提高，等权平均准确率至少提高 5 个百分点，且配对重采样的 95% 区间下限大于零。通过后在 128 组按正式的 24 轮预算复核，要求准确率至少提高 3 个百分点，AUROC 同时提高。通过复核后，才用完整的 6,400 张训练图跑两个种子并导出 CSV。 [运行记录](runs/deep-fusion/round-01/summary.json)

```powershell
python -m lqy.deep_fusion --device xpu --output lqy/runs/deep-fusion/round-01
```

两组均完成 12 轮筛选。512 组准确率从 71.43% 降到 58.97%，1024 组从 70.28% 降到 55.19%。等权平均变化为 −13.78 个百分点，配对重采样的 95% 区间为 −17.67 至 −10.03 个百分点。试验在筛选阶段停止。512 组的全图分支在抽样的训练集上为 91.86%，在整组留出验证上为 49.34%。 [训练日志](runs/deep-fusion/round-01/gated.stdout.log) · [分支诊断](runs/deep-fusion/round-01/diagnostics/branch_512.json)
