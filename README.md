# AI 生成图像识别

使用比赛标注训练真假图像二分类器，真实图像为 `0`，AI 生成图像为 `1`。各模型从随机初始化开始，只使用比赛训练子集学习。数据目录、标签和提交模板见[项目与数据说明](doc/项目与数据说明.md)。

## 三人分工

| 成员 | 先跑 | 再跑 | 这组能回答什么 |
| --- | --- | --- | --- |
| **A · 经典 CNN**（[lqy](lqy/需求文档.md)） | **ResNet-18**，标准残差卷积基线 | **MobileNetV2**，用深度可分离卷积和倒残差块降低计算量 | 在只有 6,400 张训练图时，轻量 CNN 能保留多少检测性能？ [ResNet-18 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) · [MobileNetV2 论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) |
| **B · 现代注意力**（[ddc](ddc/需求文档.md)） | **TinyNeXt-T**，ICCV 2025，约 **100 万参数** | **SHViT-S1**，CVPR 2024，约 **630 万参数** | 两者都用单头注意力，但组织卷积、下采样和注意力的方式不同。比较在这份数据上，较小的 TinyNeXt-T 能否达到同等验证表现。 [TinyNeXt](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf) · [SHViT](https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html) |
| **C · 现代状态空间**（[fqm](fqm/需求文档.md)） | **EfficientViM-M1**，CVPR 2025，约 **670 万参数**。在压缩后的隐藏状态上做特征混合，有现成的纯 PyTorch 实现 | **小型双向 Vision Mamba**，沿用此前拟定的 4 块、宽度 96 配置，正反向扫描图块 | 比较现成的分层状态空间骨干和小型双向扫描模型，哪种在短图块序列、小训练集上更有竞争力。 [EfficientViM 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_EfficientViM_Efficient_Vision_Mamba_with_Hidden_State_Mixer_based_State_CVPR_2025_paper.pdf) · [纯 PyTorch 实现说明](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models.mdx) · [Vision Mamba](https://proceedings.mlr.press/v235/zhu24f.html) |

## 统一实验协议

1. **划分与输入。** 共用按类别分层的[固定划分清单](splits/train_val.csv)。按 `split` 列读取，`train` 为训练 6,400 张，`val` 为验证 1,600 张，每类各留 800 张验证。三人共用 RGB 输入尺寸、基础预处理、增强和训练预算，归一化统计仅由训练子集估计。先划分再增强，同一原图的变体归入同一划分。文件路径、文件名和扩展名只用于加载。
2. **初筛与复跑。** 六个模型先各跑一个种子。每人从本路线选一个配置，再用三个共同约定的训练种子复跑。记录有效批量、模型参数量、训练配置、峰值显存和每轮耗时。设备不同的耗时分别列出。
3. **统一验证。** 对同一批验证图报告原图与统一质量 JPEG 重编码版的准确率、AUROC、两类召回率和混淆矩阵。训练集中真实图全部为 JPEG、生成图全部为 PNG，重编码结果用于衡量对再次压缩的敏感度。
4. **交付。** 每人提供训练命令、配置、精简逐轮日志、最优 checkpoint 的可访问位置和结果表。测试图仅用于最终推理，按比赛模板的 ID 顺序导出。

## 讨论用论文

[补充阅读索引](doc/papers/README.md)收录局部残差、频谱检测、数据偏差和架构对照的论文。
