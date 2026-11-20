# AI 生成图像检测补充阅读

| 想讨论的问题 | 论文及读法 |
| --- | --- |
| 空间域的局部信息有什么用？ | [NPR，CVPR 2024](NPR-CVPR2024.pdf)用相邻像素残差描述上采样痕迹，适合作为 RGB 基线的输入对照。 |
| 频率线索如何进入网络？ | [FreqNet，AAAI 2024（arXiv 版）](FreqNet-AAAI2024.pdf)在 CNN 内学习频域特征。[SPAI，CVPR 2025](SPAI-CVPR2025.pdf)用预训练网络学习频谱，可参考其预训练消融。 |
| 数据偏差会怎样影响验证？ | [Fake or JPEG?，ECCV 2024 Workshop（arXiv 版）](Fake-or-JPEG-arXiv.pdf)分析 JPEG 与尺寸偏差。[Bias-Free Training，CVPR 2025](Bias-Free-Training-CVPR2025.pdf)与[Aligned Datasets，ICLR 2025](Aligned-Datasets-ICLR2025.pdf)提供真假样本对齐的实验设计。 |
| 图像分类中的状态空间模块贡献有多大？ | [MambaOut，CVPR 2025](MambaOut-CVPR2025.pdf)用门控 CNN 与视觉 Mamba 对照，可参考其实验如何检验状态空间模块。 |
| 如何测试跨生成器泛化？ | [GenImage，NeurIPS 2023](GenImage-NeurIPS2023.pdf)分别设计了跨生成器和图像退化测试。若后续能获得生成器标签，可参考它按生成器留出验证集。 |
| 训练数据的生成器多样性有什么影响？ | [Community Forensics，CVPR 2025](Community-Forensics-CVPR2025.pdf)比较训练生成器的数量与多样性。[GAPL，CVPR 2026](GAPL-CVPR2026.pdf)研究扩大来源后出现的特征混叠，用生成器感知原型和 LoRA 适配预训练编码器。两篇可用于讨论已有训练数据能支持多强的泛化结论。 |
| JPEG 压缩测试还能怎样扩展？ | [RRDataset，ICCV 2025](RRDataset-ICCV2025.pdf)评估社交平台转发和再数字化对检测的影响，可参考它在相同图像上比较处理前后两类召回率的做法。 |
| 除了 NPR 和频谱，还能看什么像素线索？ | [LOTA，ICCV 2025](LOTA-ICCV2025.pdf)从低位平面提取局部信号。可借鉴确定性的位平面变换，用随机初始化的轻量 CNN 对比 RGB 输入与低位平面输入。 |
