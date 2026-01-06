# （一）多层感知机

多层感知机是最基础的神经网络模型。设数据集$\mathcal{D = \{}x,y\}$。其中$x \in \mathbb{R}^{N \times d}$为样本特征，$N$为样本数量，$d$为特征维度。$y \in \mathbb{R}^{N}$为样本类别标签，$y_{i} \in \{ 1,2,3,\ldots,C\}$，$C$为类别个数。设MLP共有$k$层，则MLP的结构如下：

$$z^{(1)} = xW^{(1)} + b^{(1)}$$

$$a^{(1)} = ReLU(z^{(1)})$$

$$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$$

$$a^{(2)} = ReLU(z^{(2)})$$

$$\ldots$$

$$z^{(k)} = a^{(k - 1)}W^{(k)} + b^{(k)}$$

$$\widehat{y} = a^{(k)} = softmax(z^{(k)})$$

其中，ReLU为隐藏层激活函数：

$$ReLU\left( z_{ij} \right) = \max\left( 0,z_{ij} \right) = \left\{ \begin{array}{r}
z_{ij},z_{ij} > 0 \\
0,z_{ij} \leq 0
\end{array} \right.\ $$

若不引入ReLU激活函数，则：

$$z^{(1)} = xW^{(1)} + b^{(1)}$$

$$z^{(2)} = z^{(1)}W^{(2)} + b^{(2)}$$

则有：

$$z^{(2)} = z^{(1)}W^{(2)} + b^{(2)} = \left( xW^{(1)} + b^{(1)} \right)W^{(2)} + b^{(2)} = xW^{(1)}W^{(2)} + \left( b^{(1)}W^{(2)} + b^{(2)} \right)$$

设$W^{*} = W^{(1)}W^{(2)},b^{*} = b^{(1)}W^{(2)} + b^{(2)}$
，则$z^{(2)} = xW^{*} + b^{*}$。由此可知，当不引入激活函数时，多层神经网络的效果相当于单层神经网络。因此隐藏层的激活函数是必要的。

softmax为输出层激活函数：

$$softmax\left( z_{ij} \right) = \frac{exp(z_{ij})}{\sum_{t}^{}{exp(z_{it})}}$$

对于分类问题，输出层激活函数用于将每个样本对应的输出向量转化为类别预测概率分布$\widehat{y} \in \mathbb{R}^{N \times C}$。损失函数采用交叉熵损失：

$$\mathcal{L = -}\frac{1}{N}\sum_{i = 1}^{N}{\log a_{iy_{i}}^{(k)}}$$

以两层神经网络为例：

$$z^{(1)} = xW^{(1)} + b^{(1)}$$

$$a^{(1)} = ReLU(z^{(1)})$$

$$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$$

$$a^{(2)} = softmax(z^{(2)})$$

根据复合函数的链式求导法则，$\frac{\partial\mathcal{L}}{\partial W^{(2)}}$，$\frac{\partial\mathcal{L}}{\partial b^{(2)}}$，$\frac{\partial\mathcal{L}}{\partial a^{(1)}}$均可以由$\frac{\partial\mathcal{L}}{\partial z^{(2)}}$进一步求得，$\frac{\partial\mathcal{L}}{\partial W^{(1)}}$，$\frac{\partial\mathcal{L}}{\partial b^{(1)}}$均可以通过$\frac{\partial\mathcal{L}}{\partial z^{(1)}}$进一步求得。因此不必从头开始推导损失函数对每个参数的梯度，而是可以先求出$\frac{\partial\mathcal{L}}{\partial z^{(i)}}$，再根据$\frac{\partial\mathcal{L}}{\partial z^{(i)}}$求$\frac{\partial\mathcal{L}}{\partial W^{(i)}}$和$\frac{\partial\mathcal{L}}{\partial b^{(i)}}$。这种求解梯度的算法称为反向传播算法。下面利用反向传播算法求解$\mathcal{L}$的梯度。

$$\mathcal{L = -}\frac{1}{N}\sum_{i = 1}^{N}{\log a_{iy_{i}}^{(2)}} = - \frac{1}{N}\sum_{i = 1}^{N}{\log\frac{\exp\left( z_{iy_{i}}^{(2)} \right)}{\sum_{j = 1}^{C}{\exp\left( z_{ij}^{(2)} \right)}}} = \frac{1}{N}\sum_{i = 1}^{N}\left( \log{\sum_{j = 1}^{C}{\exp\left( z_{ij}^{(2)} \right)}} - z_{iy_{i}}^{(2)} \right)$$

为简便计算，我们暂时不考虑$\mathcal{L}$前面的系数$\frac{1}{N}$。令$\mathcal{L}' = N\mathcal{L}$。

$$\mathcal{L}' = \sum_{i = 1}^{N}\left( \log{\sum_{j = 1}^{C}{\exp\left( z_{ij}^{(2)} \right)}} - z_{iy_{i}}^{(2)} \right)$$

则

$$\frac{\partial\mathcal{L}'}{\partial z_{uv}^{(2)}} = \frac{\partial}{\partial z_{uv}^{(2)}}\left( \log{\sum_{j = 1}^{C}{\exp\left( z_{uj}^{(2)} \right)}} - z_{uy_{u}}^{(2)} \right) = \frac{\exp\left( z_{uv}^{(2)} \right)}{\sum_{j = 1}^{C}{\exp\left( z_{uj}^{(2)} \right)}} - \left\{ \begin{array}{r}
1,v = y_{u} \\
0,v \neq y_{u}
\end{array} \right.\  = a_{uv}^{(2)} - \left\{ \begin{array}{r}
1,v = y_{u} \\
0,v \neq y_{u}
\end{array} \right.\ $$

如果标签$y$为独热编码，则$\frac{\partial\mathcal{L}'}{\partial z_{uv}^{(2)}} = a_{uv}^{(2)} - y_{uv}$，则$\delta^{z^{(2)}} = \frac{\partial\mathcal{L}'}{\partial z^{(2)}} = a^{(2)} - y$。

由$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$，可知

$$
\frac{\partial z_{st}^{(2)}}{\partial W_{uv}^{(2)}} =
\begin{cases}
a_{su}^{(1)}, & t = v, \\
0, & t \neq v.
\end{cases}
$$

则

$$\frac{\partial\mathcal{L}'}{\partial W_{uv}^{(2)}} = \sum_{i = 1}^{N}{\frac{\partial\mathcal{L}'}{\partial z_{iv}^{(2)}}\frac{\partial z_{iv}^{(2)}}{\partial W_{uv}^{(2)}}\ } = \sum_{i = 1}^{N}{\frac{\partial\mathcal{L}'}{\partial z_{iv}^{(2)}}a_{iu}^{(1)}} = \sum_{i = 1}^{N}{\delta_{iv}^{z^{(2)}}a_{iu}^{(1)}} = \sum_{i = 1}^{N}{\left( \delta^{z^{(2)}} \right)_{vi}^{T}a_{iu}^{(1)}}$$

则

$$\frac{\partial\mathcal{L}'}{\partial\left( W^{(2)} \right)_{vu}^{T}} = \sum_{i = 1}^{N}{\left( \delta^{z^{(2)}} \right)_{vi}^{T}a_{iu}^{(1)}}$$

则

$$\frac{\partial\mathcal{L}'}{\partial\left( W^{(2)} \right)^{T}} = \left( \delta^{z^{(2)}} \right)^{T}a^{(1)}$$

则

$$\frac{\partial\mathcal{L}'}{\partial W^{(2)}} = \left( a^{(1)} \right)^{T}\delta^{z^{(2)}}$$

由

$$\frac{\partial z_{st}^{(2)}}{\partial b_{v}^{(2)}} = \left\{ \begin{array}{r}
1,t = v \\
0,t \neq v
\end{array} \right.\ $$

得

$$\frac{\partial\mathcal{L}'}{\partial b_{v}^{(2)}} = \sum_{i = 1}^{N}{\frac{\partial\mathcal{L}'}{\partial z_{iv}^{(2)}}\frac{\partial z_{iv}^{(2)}}{\partial b_{v}^{(2)}}} = \sum_{i = 1}^{N}\frac{\partial\mathcal{L}'}{\partial z_{iv}^{(2)}}$$

即$\frac{\partial\mathcal{L}'}{\partial b^{(2)}}$等于$\delta^{z^{(2)}}$每列相加得到的向量。

由$z^{(2)} = a^{(1)}W^{(2)} + b^{(2)}$，得$\left( z^{(2)} \right)^{T} = \left( W^{(2)} \right)^{T}\left( a^{(1)} \right)^{T} + \left( b^{(2)} \right)^{T}$。

类比可知：

$$\frac{\partial\mathcal{L}'}{\partial\left( a^{(1)} \right)^{T}} = W^{(2)}\left( \delta^{z^{(2)}} \right)^{T}$$

则

$$\delta^{a^{(1)}} = \frac{\partial\mathcal{L}'}{\partial a^{(1)}} = \delta^{z^{(2)}}\left( W^{(2)} \right)^{T}$$

由$a^{(1)} = ReLU(z^{(1)})$，得

$$\delta^{z^{(1)}} = \frac{\partial\mathcal{L}'}{\partial z^{(1)}} = \frac{\partial\mathcal{L}'}{\partial a^{(1)}}\frac{\partial ReLU\left( z^{(1)} \right)}{\partial z^{(1)}} = \delta^{a^{(1)}}\frac{\partial ReLU\left( z^{(1)} \right)}{\partial z^{(1)}}$$

由$z^{(1)} = xW^{(1)} + b^{(1)}$，得

$$\frac{\partial\mathcal{L}'}{\partial W^{(1)}} = x^{T}\delta^{z^{(1)}}$$

$$\frac{\partial\mathcal{L}'}{\partial b_{v}^{(1)}} = \sum_{i = 1}^{N}\frac{\partial\mathcal{L}'}{\partial z_{iv}^{(1)}}$$

即$\frac{\partial\mathcal{L}'}{\partial b^{(1)}}$等于$\delta^{z^{(1)}}$每列相加得到的向量。

# （二）卷积神经网络（CNN）

## 1、输入层：
原始图像（channels \* height \* width；channels为通道数量，通常为RGB三通道；height为图像高度；width为图像宽度）。

## 2、特征学习阶段：

### （1）卷积层
卷积层是特征提取的核心。这是CNN最核心、最创新的部分。

- **卷积核：**
  它是一个小尺寸的权重矩阵（如3 \* 3, 5 \* 5）。卷积核以滑动窗口的方式，系统地遍历输入图像的每一个局部区域。一个卷积层通常有多个卷积核，因此会输出多个特征图，每张特征图提取了输入数据的不同特征。每个卷积核专门负责探测一种特征。比如，有的卷积负责探测边缘，有的负责探测颜色，有的负责探测纹理。

- **卷积操作：**
  卷积核在输入图像或上一层的特征图上从左到右、从上到下滑动。在每一个位置，卷积核与输入图像的对应区域进行点乘并求和，得到一个单一的数值。这个滑动和计算的过程，就生成了一个特征图。

- **卷积的步长：** 滤波器每次滑动的像素数。步长越大，输出特征图尺寸越小。

- **填充：** 有时在输入图像周围填充一圈0，可以控制输出特征图的尺寸，这样做通常是为了保持尺寸不变。

- **权重共享：** 不同于MLP中特征向量的每个维度对应一个权重，CNN中利用一个卷积核扫描整张图像，即一个卷积核的权重在整张图像中共享，减少了参数量，提高了计算效率。

### （2）激活函数
卷积操作是线性的。如（一）中所述，如果不引入非线性激活函数，多个线性层效果相当于一个线性层。常用ReLU作为激活函数。

### （3）池化层
池化层通常跟在卷积层和激活函数之后。其将图像划分成一个个固定大小的patch，每个patch取最大值或平均值。其目的如下：

- **降维：** 减小特征图的尺寸，从而减少计算量和参数。
- **防止过拟合：** 通过降维，提供了一种抽象表示。

有两种池化方式：

- **最大池化：** 取窗口内的最大值。这是最常用的方法，它能保留最显著的特征。
- **平均池化：** 取窗口内的平均值。

池化不包含需要学习的参数，它只是一个固定的操作。

## 3、分类决策阶段（全连接层）：
将卷积和池化得到的特征图展平，通过线性全连接层得到一个维度等于类别个数的向量，用于分类决策。

## 4、输出层：
通常使用 softmax 激活函数，将最终输出转换为每个类别的概率。

# （三）思考与延伸

## 1. 性能与效率：
### (1) MLP 和 CNN 哪个模型在 MNIST 任务上表现更好？为什么？
CNN在MNIST任务上明显表现更好。原因如下：
MLP仅仅机械地将二维图像展平，当图像像素数量很高时，输入向量的维度会非常高，神经网络第一层需要的参数数量也非常多，增大了计算量，而CNN通过权重共享和池化操作减少了参数数量，提高了计算效率。
此外，图像并不仅仅是像素值的集合，不同像素之间的位置关系也非常重要。由于MLP仅仅机械地将二维图像展平，导致其忽略了图像的空间结构，完全破坏了像素之间的上、下、左、右、邻近等位置关系。而CNN的卷积核在对图像的每个局部区域进行扫描时，有效保留了每个局部区域内的像素之间的邻近位置关系信息，且由于点积得到的值在特征图中的位置与局部区域在原图中的位置相同，使得在第二轮卷积操作中，不同局部区域之间的邻近位置关系信息也得到保留。
综合上述原因，CNN比MLP在MNIST任务上表现更好。

### (2) 计算并比较两个模型的总参数数量。为什么 CNN 在参数更少的情况下，通常能达到更高的精度？
MLP：
W1: 784 * (784 / 4) = 153664
b1: 784 / 4 = 196
W2: 784 / 4 * 10 = 1960
b2: 10
total: 155830

CNN:
Conv1: 16 * 5 * 5 = 400
Conv2: 32 * 16 * 5 * 5 = 12800
FC: 7 * 7 * 32 * 10 + 10 = 15690
total: 28890

可见CNN的参数量比MLP减少了约80%。

CNN参数更少是由于其权重共享和池化操作。CNN的卷积核在对图像的每个局部区域进行扫描时，有效保留了每个局部区域内的像素之间的邻近位置关系信息，且由于点积得到的值在特征图中的位置与局部区域在原图中的位置相同，使得在第二轮卷积操作中，不同局部区域之间的邻近位置关系信息也得到保留。而MLP只是机械地将原图展平为一维向量，破坏了原图的位置关系信息。因此CNN在参数更少的情况下可以达到更高的精度。

## 2. 数据处理的哲学：
### (1) MLP 的“扁平化”操作丢失了图像的什么重要信息？
丢失了图像像素之间的上、下、左、右、邻近等位置关系信息。

### (2) CNN 是如何通过其结构（卷积核）来利用这些信息的？
CNN的卷积核在对图像的每个局部区域进行扫描时，有效保留了每个局部区域内的像素之间的邻近位置关系信息，且由于点积得到的值在特征图中的位置与局部区域在原图中的位置相同，使得在第二轮卷积操作中，不同局部区域之间的邻近位置关系信息也得到保留。

## 3. 模型的泛化能力：
想象一下，如果测试集中的数字被轻微平移或旋转，哪个模型（MLP 还是 CNN）的性能下降会更小？为什么？
CNN的性能下降会更小。
由于MLP直接将图像展平为一维向量，如果测试集中的数字被平移或旋转，这个一维向量的每个位置的元素之间的大小关系就会明显发生改变，这个向量的方向就会显著变化。当这个变化后的向量与同样的模型权重相乘时，就很容易导致模型对其错误分类。
CNN则是利用一个相同的卷积核扫描整张图像，因此无论数字在图像的什么位置，其都存在于这个图像中，且像素值的局部位置关系未发生改变，因此都能被这个卷积核扫描到。此外，由于CNN的池化操作将特征图的局部像素值聚集到一起，使得池化得到的特征图中每个像素值包含了原图像中更大范围的信息，从而模型对位置变化的敏感度下降。因此模型性能不会发生明显下降。

## 4. 应用场景：
如果任务是根据用户的年龄、收入、职业等表格数据来预测其信用等级，你会选择 MLP 还是 CNN？请说明理由。
我会选择MLP。用户的年龄、收入、职业等数据分别代表了该用户在不同维度上的特征，这些特征之间并不具备如空间位置那样的关联性。MLP适用于处理这类彼此独立、无相关性的特征，能够有效提取每个特征的信息。而CNN则倾向于假设输入数据具有局部相关性，若将其应用于此类非空间结构数据，相当于人为引入并不存在的关联，可能导致模型对特征关系产生错误理解。