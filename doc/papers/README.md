# AI 生成图像检测补充阅读

| 想讨论的问题 | 论文及读法 |
| --- | --- |
| 空间域的局部信息有什么用？ | [NPR，CVPR 2024](NPR-CVPR2024.pdf)用相邻像素残差描述上采样痕迹，适合作为 RGB 基线的输入对照。 |
| 频率线索如何进入网络？ | [FreqNet，AAAI 2024（arXiv 版）](FreqNet-AAAI2024.pdf)在 CNN 内学习频域特征。[SPAI，CVPR 2025](SPAI-CVPR2025.pdf)用预训练网络学习频谱，可参考其预训练消融。 |
| 数据偏差会怎样影响验证？ | [Fake or JPEG?，ECCV 2024 Workshop（arXiv 版）](Fake-or-JPEG-arXiv.pdf)分析 JPEG 与尺寸偏差。[Bias-Free Training，CVPR 2025](Bias-Free-Training-CVPR2025.pdf)与[Aligned Datasets，ICLR 2025](Aligned-Datasets-ICLR2025.pdf)提供真假样本对齐的实验设计。 |
| 图像分类中的状态空间模块贡献有多大？ | [MambaOut，CVPR 2025](MambaOut-CVPR2025.pdf)用门控 CNN 与视觉 Mamba 对照，可参考其实验如何检验状态空间模块。 |
