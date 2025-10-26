# Introduction: Deep Learning and Machine Learning Basics

* 深度学习和机器学习的关系与区别
  * 深度学习通过多层非线性变换自动学习数据表征（如CNN卷积层逐层提取边缘→纹理→语义），传统机器学习依赖人工设计特征
  * 深度学习通过深度神经网络的高容量拟合复杂模式，但需通过正则化（L2、Dropout）和对抗训练防止过拟合
  * 深度学习需要海量标注数据（因为要学习非常深层的知识），而传统机器学习在小数据表现更优

* 数据表示（特征 vs. 表征学习）
  * 特征空间层次化：像素级原始数据→浅层特征（边缘）→中层特征（物体部件）→高层语义（类别）
  * 分布式表示：每个神经元编码局部特征组合
* 基本模型：线性回归、逻辑回归、损失函数（MSE、交叉熵）
  * MSE损失： $L=\frac{1}{N}\sum^N_{i=1}(y_i-\hat{y}_i)^2$ ，闭式解 $w=(X^TX)^{-1}X^Ty$
  * 逻辑回归： $\sigma(z)=\frac{1}{1+e^{-z}}$ ，即输出属于正类的概率，交叉熵损失 $L = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log(1 - \sigma(\hat{y}_i))$

# Optimization and Loss Functions

* Adam优化器
  * 动量与自适应学习率结合 $m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$ ， $\quad v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$ ， $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ ， $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ ， $\Delta \mathbf{w} = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ ，超参数设置： $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1e - 8$ 

* 梯度消失与爆炸的解决
  * ResNet的残差连接： $y_l=x_l+F(x_l,W_l)$ ，反向传播时梯度可直接通过恒等映射传递，避免梯度消失

# Perceptron and Neural Network

* 感知机模型与前馈神经网络结构
* 激活函数（Sigmoid、ReLU、Tanh、LeakyReLU）
  * ReLU: $f(x)=\max (0,x)$ ，解决梯度消失问题，但存在dead ReLU问题
  * Leaky ReLU: $f(x)=\max (0.01x,x)$ ，缓解Dead ReLU
* 多层感知机（MLP）与非线性建模能力
* 参数初始化
  * Xavier初始化：标准差 $=\sqrt{\frac{2}{n_{in}+n_{out}}}$ ，适用于Sigmoid/Tanh激活函数
  * He初始化 $=\sqrt{\frac{2}{n_{in}}}$ ，适用于ReLU及其变体
* 正则化
  * L2正则化： $L=L_{CE}+\frac{\lambda}{2}|w|^2_2$ ，防止过拟合，使权重趋向于0
  * Dropout：训练时随机失活神经元，如以0.5概率关闭隐藏层神经元，测试时使用全连接但权重乘以0.5

# Convolutional Neural Network Architectures

* 卷积操作、池化、填充、步幅
  * 
* CNN 典型结构：LeNet、AlexNet、VGG、ResNet
* 特征图、感受野、参数共享
* 可视化卷积滤波器与特征图

# Course Project 1: Image Classification with Convolutional Neural Networks

* 使用 PyTorch/TensorFlow 构建 CNN 模型
* 数据预处理、增强、批量训练
* 模型评估（准确率、混淆矩阵、Top-k 准确率）
* 调参技巧与训练监控（TensorBoard）


# Efficient Convolutional Neural Networks

* 模型压缩与加速：剪枝、量化、知识蒸馏
* 轻量级网络：MobileNet、ShuffleNet、SqueezeNet
* 深度可分离卷积、组卷积

# Unsupervised Deep Learning

* 自编码器（AE）、稀疏自编码器、去噪自编码器
* 变分自编码器（VAE）
* 无监督预训练与表征学习
* 聚类与深度嵌入（如 DEC）

# Applications in Computer Vision and Data Analytics

* 图像分类、目标检测、语义分割简介
* 深度特征提取与迁移学习
* 可视化分析（t-SNE、CAM、Grad-CAM）
* 深度学习在医疗影像、遥感、工业检测中的应用案例

# Course Project 2: Image Segmentation with MobileNets

* 使用轻量级网络进行语义分割
* 掌握分割任务的评价指标（IoU、Dice 系数）
* 数据标注工具与分割数据集（如 Pascal VOC、Cityscapes）

# Recurrent Neural Networks and LSTM

* RNN 的结构与梯度消失/爆炸问题
* LSTM、GRU 的门控机制
* 序列到序列模型（Seq2Seq）
* 应用：文本生成、时间序列预测、语音识别

# Transformer and Its Recent Variations

* 自注意力机制（Self-Attention）
* Transformer 架构（Encoder-Decoder、Positional Encoding）
* 多头注意力、层归一化、残差连接
* 后续变体：BERT、GPT、T5、ViT（Vision Transformer）

# Course Project 3: Image Classification with Vision Transformer

* 使用 ViT 进行图像分类
* 理解图像块嵌入（Patch Embedding）、位置编码
* 与 CNN 性能对比，分析适用场景

# MambaandLinearBig Model

* 状态空间模型（SSM）与 Mamba 架构
* 线性注意力机制与高效长序列建模
* 与 Transformer 的复杂度对比（O(n²) vs. O(n)）

# Generative Models

* 生成对抗网络（GAN）：Generator vs. Discriminator
* 深度生成模型：VAE、GAN、Diffusion Models
* 训练技巧与常见问题（模式崩塌、训练不稳定）
* 应用：图像生成、风格迁移、数据增强

# Course Project 4: Generative Models and Applications

* 实现一个 GAN 或 VAE 模型
* 生成图像样本并评估质量（FID、IS 指标）
* 探索条件生成（Conditional GAN）

# Multimodal Deep Learning

* 多模态融合策略（早期、晚期、联合融合）
* 图文模型：CLIP、BLIP、ALBEF
* 应用：图像字幕生成、视觉问答（VQA）、文本图像检索

# Adversarial Deep Learning

* 对抗样本与攻击方法（FGSM、PGD）
* 模型鲁棒性与防御机制（对抗训练、输入变换）
* 安全性与可信 AI 的基本概念

 
