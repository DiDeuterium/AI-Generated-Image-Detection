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
