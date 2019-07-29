## 5MB的神经网络也高效，Facebook新压缩算法造福嵌入式设备

原创： 关注前沿科技 [量子位](javascript:void(0);) *昨天*

##### 鱼羊 发自 凹非寺  量子位 报道 | 公众号 QbitAI

人工智能风暴袭来，机器人、自动驾驶汽车这样的嵌入式设备也热度渐长。毫无疑问，现在，嵌入式设备也需要高效的神经网络加持。

但是，如何在嵌入式设备上实现高效的神经网络，可不是一件简单的事情。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSwC7D5d4KFqWiczNeBQ2ibxMiaAUV7FIVg8GIksbl3yMSnb4YXsrAERGYg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

性能，功耗，成本，都是不得不考虑的问题。每一种不同的应用场景，都需要在神经网络的大小和精确度之间进行特定的权衡（trade-off）。

像自动驾驶汽车，就需要实现对实时视频的精确识别，这意味着嵌入其中的神经网络模型一定是个体积庞大的家伙。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tStfFq9Icia38TOgpvribNUibMEDxhavnAvVDTvcTSMibIyibGibgaUtTKaRGA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

受限于硬件，神经网络必须进行压缩。

在卷积神经网络压缩这个课题上，移动高效架构是主流（MobileNets或ShuffleNets）。

然而，基于移动高效架构的MobileNet-v2在ImageNet对象分类中虽然已经达到了**71%**的 top-1准确率，但这仍远落后于卷积神经网络的最佳表现**83.1%**。

Facebook的研究人员们决定转换思路，既然如此，何不更专注于传统的卷积网络本身呢？

## 重新审视神经网络的量化

Facebook提出了一种适用于ResNet类架构的压缩方法，名叫**Bit Goes Down**。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tScT3fxk36I0kMnQ1QOl2xCYX920hXnryWkUQywqlGTSmnsrFa5icEGAA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这种方法利用了**结构化量化算法PQ（Product Quantization）**中卷积的高相关性，重点关注activations的重建，而不是权重本身。

也就是说，这种方法只关注域内输入的重建质量。

研究人员让未经压缩的神经网络充当“老师”，利用图灵奖得主Hinton等人提出的提炼（distillation）技术来指导“学生”网络的压缩。

这一过程无需任何标记数据，是一种无监督学习方法。

具体的实现方法是这样的：

#### 一、层量化

先以**全连接层**为例。

PQ算法的任务是量化全连接层的权重矩阵。但从实际需求来看，权重不重要，保留层的输出才是研究人员的关注重点。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSFhFTTJ2lbhpSsDDarwA3x4bIRcz43tsciazW2o2yqc2SjLdmMlo66Ag/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在上图这种非线性分类器中，不能保证层权重的Frobenius近似是某个任意域输出的最佳近似（特别是对于域内输入）。

因此，研究人员提出了一个替代方案，通过将层应用于域内输入获得输出激活（activations）的重建误差，直接最小化该误差。用编码簿（codebook）最大限度地减少输出激活及其重建之间的差异。

接着要对EM算法（最大期望化算法）进行调整。

E-step是集群分配，这一步通过详尽的探索来执行。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tS3Szn7FfjVmekOEic0GiccticfyG8mXib0JQzf1iaM3u4kEjmd5xl05tQzPg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

M-step是码字（codeword）更新，这一步通过显式计算最小二乘问题的解来完成，实际上就是在E-step和M-step交替之前，计算x tilde的伪逆。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSvWRy81Eb4jjwSxA0ficfF45vXCXOY35MG6czjGdZiaUrdgarPMa5wlibA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

需要注意的是在E-step之后某些集群可能为空。

对于**卷积层**情况又是如何呢？

在完全连接层，这一方法适用于任何矢量集，所以如果将相关的4D权重矩阵分割成一组向量，该方法就可以应用于卷积层。

分割4D矩阵的方法有很多，标准就是要最大化矢量之间的相关性，因为当矢量高度相关时，基于矢量量化的方法效果最好。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSDppS48Vah04wtJVSyk8k3OibiaxbtCPG7LcyQJ8iaCgXUeXXTzFDDDSQQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

就像这样，在空间上量化卷积滤波器以利用网络中的信息冗余，不同颜色代表拥有不同码字的子向量。

#### 二、网络量化

接下来，就涉及到对整个神经网络的量化。

首先，这是**自下而上的量化**，从最低层开始，到最高层结束。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tS90FE0hiaC21V8zX2y7DeauNWkDPxTCWCou2PVgtU4rzR8dVyziccVJPQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这也就是所谓的用非压缩的教师网络引导学生网络的压缩。主要包括以下两个步骤：

**学习码字**

恢复该层的**当前（current）**输入激活，即通过量化后的低层转发一批图像而获得的输入激活。使用这些激活量化当前层。

**微调码字**

采用Hinton的distillation方法微调码字，以非压缩网络作为教师网络，当前层之前的压缩网络作为学生网络。

在这一步骤中，通过对分配给指定码字的每个子矢量的梯度求平均，来完成码字的精细化。更确切地说，是在量化步骤之后，一次性修复分配。

接下来，就剩下最后一步，**全局微调所有层的码字**，以减少残余漂移。同时更新BatchNorm层的统计数据。

全局微调使用的是标准的ImageNet训练集。

## 小体积，高精度

研究人员用Bit Goes Down量化了在ImageNet数据集上预先训练好的vanila ResNet-18和ResNet-50。

在16GB的 Volta V100 GPU上跑了一天之后，终于到了展示成果的时候。

首先，是跟标准ResNet-18和ResNet-50的比较。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSrQMnQ7M5iaQBfCCMVP7CRb8icicfMqtCJ9SyOo2ogSqcVUPc5icD9N5geg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于ResNet-18，在29倍压缩的情况下，模型大小缩小到了1.54MB，而top-1准确率仅比标准模型降低了不到4%。

ResNet-50上模型大小略大一些，但也达到了5MB左右，准确率同样保持在一个可以接受的水平。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tScNyMhnPnuvfuvic2ibjX1CFXg2MGAWg4JzUhwJhzfWibn50gzcTvlia93Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

跟模型压缩界的前辈相比，Bit Goes Down表现出了它的优势，虽然在1MB的指定大小中败下阵来，但在5MB的比拼中，新方法优势明显，准确率提升了将近5个百分点。

这意味着压缩后的模型获得了非压缩ResNet-50的性能，同时还只有5MB大小。

Bit Goes Down在图像分类上表现不俗，在图像检测方面又如何呢？

研究团队又压缩了何恺明的Mask R-CNN。这回用上了8块V100 GPU来进行训练。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSe4KwOCWAdCfaX3sicyicQl54KFgXibibXtsIuIxaK7eJSeaXh4NiaZaSM8w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在压缩了25倍的情况下，压缩模型的Box AP和Mask AP都只下降了4左右。

这表现，着实有些厉害。

Facebook表示，Bit Goes Down这样的压缩算法将推动虚拟现实（VR）和增强现实（AR）等技术的进一步发展。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtASA0WZj8Dgjcm3gVyDV7tSXLQnKLH7XsIwv6ic4OwQJyLJQVzUFib2LlCIrN9evCoGxWAiauD7f2q6Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

目前，研究团队已经开源了压缩模型及代码，如果感兴趣，你可以亲自复现一下～

## 传送门

博客地址：
https://ai.facebook.com/blog/compressing-neural-networks-for-image-classification-and-detection/

论文地址：
https://arxiv.org/abs/1907.05686

GitHub地址：
https://github.com/facebookresearch/kill-the-bits