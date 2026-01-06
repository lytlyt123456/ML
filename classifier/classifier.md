# （一）支持向量机算法 (Support Vector Machine, SVM)

## 1、Basic SVM------解决线性可分的二分类问题

对于线性可分的二分类数据集$\mathcal{D =}\left\{ x_{i},y_{i} \right\}_{i = 1}^{N}$，其中$y_{i} \in \left\{ - 1,\  + 1 \right\}$。训练目标为找到一个超平面

$$\omega^{T}x + b = 0$$

将数据集$\mathcal{D}$的两个类别分隔开：

- 对于标签为--1的样本$x_{i}$，在超平面的一侧，满足$\omega^{T}x_{i} + b < 0$；
- 对于标签为+1的样本$x_{i}$，在超平面的另一侧，满足$\omega^{T}x_{i} + b > 0$。

即：对每个样本，满足$y_{i}\left( \omega^{T}x_{i} + b \right) > 0$。

向标签为+1的样本一侧平移超平面，当第一个样本恰好落在其上时，记该超平面为$\omega^{T}x + b = t_{1}$，其中$t_{1} > 0$；向标签为--1的样本一侧平移超平面，当第一个样本恰好落在其上时，记该超平面为$\omega^{T}x + b = - t_{2}$，其中$t_{2} > 0$。

除了使得超平面分开两类外，我们还希望：

- $\omega^{T}x + b = 0$恰好位于超平面$\omega^{T}x + b = t_{1}$和$\omega^{T}x + b = - t_{2}$的中间，即$t_{1} = t_{2}$。记$t_{1} = t_{2} = t$。
- 超平面$\omega^{T}x + b = t$和$\omega^{T}x + b = - t$的间距尽可能大，即对$\omega^{T}x + b = t$上的点$x_{0}$，最大化

$$d = \frac{|\omega^{T}x_{0} + b + t|}{{||\omega||}_{2}} = \frac{|t + t|}{{||\omega||}_{2}} = \frac{2|t|}{{||\omega||}_{2}}$$

我们的目标是优化超参数$\omega$和$b$，但这里还有未知参数$t$存在，因此令$t = 1$，则$d = \frac{2}{{||\omega||}_{2}}$。最大化$\frac{2}{{||\omega||}_{2}}$，即最小化$\frac{{||\omega||}_{2}}{2}$，即最小化$\frac{{||\omega||}_{2}^{2}}{2}$。由于我们令$t$的值为1，所以最初提出的分类目标应改为：对每个样本，满足$y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1$。

我们重新整理一下优化目标：  
* $\min_{\omega}\frac{{||\omega||}_{2}^{2}}{2}$；
* $\forall i \in \left\{ 1,2,\ldots,N \right\},y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1$。

我们在此过于严苛地令每个样本都满足$y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1$，可能会由于某些奇异样本的存在而影响全局的分类效果。应当适当地进行松弛。因此引入松弛变量$\left\{ \xi_{i} \right\}_{i = 1}^{N}$，其中$0 \leq \xi_{i} < 1$。我们希望$\xi_{i}$的值尽可能小。

引入松弛变量后，我们的训练目标变为：  
* $\min_{\omega,\xi}{\frac{{||\omega||}_{2}^{2}}{2} + \frac{C}{n}\sum_{i = 1}^{N}\xi_{i}}$，其中$C$为超参数，称为惩罚因子 (penalty term)。
* $\forall i \in \left\{ 1,2,\ldots,N \right\},y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1 - \xi_{i}$。
* $\forall i \in \left\{ 1,2,\ldots,N \right\},0 \leq \xi_{i} < 1$。

为简化求解过程，我们需要将上述约束最优化问题转化为无约束最优化问题。

- 当满足$y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1$时，不需要松弛变量的存在，此时可令$\xi_{i} = 0$；
- 当$y_{i}\left( \omega^{T}x_{i} + b \right) < 1$时，需要松弛变量的存在，此时$\xi_{i} \geq 1 - y_{i}\left( \omega^{T}x_{i} + b \right)$，由于我们希望$\xi_{i}$尽可能小，所以我们另$\xi_{i} = 1 - y_{i}\left( \omega^{T}x_{i} + b \right)$。

即：$\xi_{i} = max\{ 0,1 - y_{i}\left( \omega^{T}x_{i} + b \right)\}$。

则我们最终的训练目标为：  
$$\min_{\omega,b}{\mathcal{L}(\omega,b)} = \min_{\omega,b}{\frac{{||\omega||}_{2}^{2}}{2} + \frac{C}{n}\sum_{i = 1}^{N}{max\{ 0,1 - y_{i}\left( \omega^{T}x_{i} + b \right)\}}}$$

下面求解梯度：  
$$\frac{\partial\mathcal{L}}{\partial\omega} = \omega + \frac{C}{N}\sum_{i = 1}^{N}\frac{\partial\xi_{i}}{\partial\omega}$$

其中  
$$\frac{\partial\xi_{i}}{\partial\omega} = \left\{ \begin{array}{r} - y_{i}x_{i},1 - y_{i}\left( \omega^{T}x_{i} + b \right) \geq 0 \\
0,1 - y_{i}\left( \omega^{T}x_{i} + b \right) < 0
\end{array} \right.\ $$

$$\frac{\partial\mathcal{L}}{\partial b} = \frac{C}{N}\sum_{i = 1}^{N}\frac{\partial\xi_{i}}{\partial b}$$

其中  
$$\frac{\partial\xi_{i}}{\partial b} = \left\{ \begin{array}{r} - y_{i},1 - y_{i}\left( \omega^{T}x_{i} + b \right) \geq 0 \\
0,1 - y_{i}\left( \omega^{T}x_{i} + b \right) < 0
\end{array} \right.\ $$

后面按梯度下降法优化$\omega$和$b$即可。

## 2、SVM的对偶问题 (Dual Problem)

回到SVM的最初优化目标和约束条件：  
* $\min_{\omega}\frac{{||\omega||}_{2}^{2}}{2}$；
* $\forall i \in \left\{ 1,2,\ldots,N \right\},y_{i}\left( \omega^{T}x_{i} + b \right) \geq 1$。

我们希望$y_{i}\left( \omega^{T}x_{i} + b \right)$尽可能大，即$1 - y_{i}\left( \omega^{T}x_{i} + b \right)$尽可能小。则我们可以定义损失函数：  
$$\mathcal{L}(\omega,b,\alpha) = \frac{{||\omega||}_{2}^{2}}{2} + \sum_{i = 1}^{N}{\alpha_{i}\left( 1 - y_{i}\left( \omega^{T}x_{i} + b \right) \right)}$$

其中$\alpha_{i}$为权重系数，满足$0 \leq \alpha_{i} \leq C$，其中$C$为惩罚因子。

在优化$\omega$和$b$之前，我们需要先找到合适的$\alpha$的值。

- 在$1 - y_{i}\left( \omega^{T}x_{i} + b \right) \ll 0$时，分类效果非常好，不需再进行优化，此时应使$\alpha_{i}$的值接近0；
- 在$1 - y_{i}\left( \omega^{T}x_{i} + b \right) \geq 0$时，说明分类效果一般或分类错误，需要进一步优化，此时需要较大的$\alpha_{i}$的值，使其在损失函数中凸显出来，从而重点进行优化。

我们希望在$1 - y_{i}\left( \omega^{T}x_{i} + b \right) < 0$时，$\alpha_{i}$尽可能小；在$1 - y_{i}\left( \omega^{T}x_{i} + b \right) \geq 0$时，$\alpha_{i}$尽可能大，即：我们希望$\alpha_{i}\left( 1 - y_{i}\left( \omega^{T}x_{i} + b \right) \right)$尽可能大。

找到合适的$\alpha$后，我们再最小化损失函数，优化$\omega$和$b$。

综上，我们的优化目标为：  
$$\min_{\omega,b}{\max_{0 \leq \alpha_{i} \leq C}{\frac{{||\omega||}_{2}^{2}}{2} + \sum_{i = 1}^{N}{\alpha_{i}\left( 1 - y_{i}\left( \omega^{T}x_{i} + b \right) \right)}}}$$

该问题的对偶问题是：  
$$\max_{0 \leq \alpha_{i} \leq C}\min_{\omega,b}{\frac{{||\omega||}_{2}^{2}}{2} + \sum_{i = 1}^{N}{\alpha_{i}\left( 1 - y_{i}\left( \omega^{T}x_{i} + b \right) \right)}}$$

在此问题中，二者等价。

**（1）** $\min_{\omega,b}{\frac{{||\omega||}_{2}^{2}}{2} + \sum_{i = 1}^{N}{\alpha_{i}\left( 1 - y_{i}\left( \omega^{T}x_{i} + b \right) \right)}}$

$$\frac{\partial\mathcal{L}}{\partial\omega} = \omega - \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}}$$

令$\frac{\partial\mathcal{L}}{\partial\omega} = 0$，则

$$\omega = \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}}$$

$$\frac{\partial\mathcal{L}}{\partial b} = - \sum_{i = 1}^{N}{y_{i}\alpha_{i}}$$

在$b$一定时， $\frac{\partial\mathcal{L}}{\partial\omega} = 0$的位置只有一个，说明最小值在$\omega = \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}}$处取得；此步骤中将$\alpha_{i}$视为常量，因此$\frac{\partial\mathcal{L}}{\partial b}$为定值，我们另$\frac{\partial\mathcal{L}}{\partial b} = 0$，则$\sum_{i = 1}^{N}{y_{i}\alpha_{i}} = 0$。

在此步骤中，我们求得使得损失函数最小的$\omega = \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}}$，并构造出一个约束条件$\sum_{i = 1}^{N}{y_{i}\alpha_{i}} = 0$。

将上式代入$\mathcal{L}(\omega,b,\alpha)$，有：  
$$\mathcal{L}(\alpha) = - \frac{1}{2}\sum_{i = 1}^{N}{\sum_{j = 1}^{N}{\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}}} + \sum_{i = 1}^{N}\alpha_{i}$$

**（2）** $\max_{0 \leq \alpha_{i} \leq C}{- \frac{1}{2}\sum_{i = 1}^{N}{\sum_{j = 1}^{N}{\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}}} + \sum_{i = 1}^{N}\alpha_{i}}$

$$\frac{\partial\mathcal{L}}{\partial\alpha_{t}} = 1 - \sum_{i = 1}^{N}{\alpha_{i}y_{i}y_{t}x_{i}^{T}x_{t}}$$

这里应注意，由于我们要最大化这个函数，所以应使用梯度上升法进行优化，而非梯度下降。

最终的优化目标为：在$0 \leq \alpha_{i} \leq C$和$\sum_{i = 1}^{N}{y_{i}\alpha_{i}} = 0$下，  
$$\max_{0 \leq \alpha_{i} \leq C}{- \frac{1}{2}\sum_{i = 1}^{N}{\sum_{j = 1}^{N}{\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}}} + \sum_{i = 1}^{N}\alpha_{i}}$$

**（3）** 在求出$\alpha$后，$\omega$由公式$\omega = \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}}$得到。

下面求解偏置$b$。对于支持向量$x_{support}$（满足$y_{i}\left( \omega^{T}x_{i} + b \right) = 1$的样本称为支持向量），设其标签为$y_{support}$。在支持向量上满足：$b = y_{support} - \omega^{T}x_{support} = y_{support} - \sum_{i = 1}^{N}{\alpha_{i}y_{i}x_{i}^{T}x_{support}}$。根据前面所述，支持向量的分类效果是所有样本中分类效果较差的，所以$\alpha_{i}$的值较大。因此实现中可以选择出$\alpha_{i}$最大的几个样本作为备选的支持向量，分别利用这些样本求出偏置$b$，再取平均即可。

**（4）** 决策函数：${\widehat{y}}_{i} = \omega^{T}x_{i} + b = \sum_{j = 1}^{N}{\alpha_{j}y_{j}x_{j}^{T}x_{i}} + b$

## 3、核函数与非线性可分问题

当样本非线性可分时，可通过映射函数$\Phi$将样本特征映射到高维向量空间。在低维向量空间中非线性可分的样本很可能在高维空间中实现线性可分。

$\Phi$的表达式涉及多个维度，通常较为复杂，但我们通常不需要知道$\Phi$的具体表达式，只需要知道两个高维向量的内积函数的表达式  
$$K\left( x_{i},x_{j} \right) = {\Phi\left( x_{i} \right)}^{T}\Phi(x_{j})$$

即可。$K\left( x_{i},x_{j} \right)$称为核函数。经常使用的核函数为高斯核函数 (RBF核函数) ：  
$$K\left( x_{i},x_{j} \right) = exp\left( - \frac{{||x_{i} - x_{j}||}_{2}^{2}}{2\sigma^{2}} \right)$$

根据对偶问题，我们的优化目标为：在$0 \leq \alpha_{i} \leq C$和$\sum_{i = 1}^{N}{y_{i}\alpha_{i}} = 0$下，  
$$\max_{0 \leq \alpha_{i} \leq C}{- \frac{1}{2}\sum_{i = 1}^{N}{\sum_{j = 1}^{N}{\alpha_{i}\alpha_{j}y_{i}y_{j}\Phi\left( x_{i} \right)^{T}\Phi(x_{j})}} + \sum_{i = 1}^{N}\alpha_{i}}$$

即  
$$\max_{0 \leq \alpha_{i} \leq C}{- \frac{1}{2}\sum_{i = 1}^{N}{\sum_{j = 1}^{N}{\alpha_{i}\alpha_{j}y_{i}y_{j}K\left( x_{i},x_{j} \right)}} + \sum_{i = 1}^{N}\alpha_{i}}$$

$$\frac{\partial\mathcal{L}}{\partial\alpha_{t}} = 1 - \sum_{i = 1}^{N}{\alpha_{i}y_{i}y_{t}K\left( x_{i},x_{t} \right)}$$

决策函数  
$$\widehat{y_{i}} = \sum_{j = 1}^{N}{\alpha_{j}y_{j}K(x_{j},x_{i})} + b$$

其中  
$$b = y_{support} - \sum_{i = 1}^{N}{\alpha_{i}y_{i}K(x_{i},x_{support})}$$

## 4、SVM实现多类分类

常见的方法为One-vs-One方法。即：

设类别个数为$k$。在$k$个类别中，利用每两个类别的训练样本训练一个SVM分类器，共训练$C_{k}^{2}$个SVM分类器。训练完毕后，决策算法如下：

一开始所有类别的票数均为0。

对测试集中的每个样本$x_{i}$，将$x_{i}$依次输入进每个分类器中。对于每个分类器的两个类别$cls_{1}$和$cls_{2}$，如果$x_{i}$被分到$cls_{1}$类别，则$cls_{1}$的票数加1，反之$cls_{2}$的票数加1。

最终，票数最多的类别为$x_{i}$的预测类别。

这种方法的缺点是：当类别个数过多时，SVM分类器的数量也会过多，带来极高的训练和推理复杂度。

# （二）逻辑回归算法 (Logistic Regression)

## 1、二分类逻辑回归算法

支持向量机算法可以较好地解决二分类问题，但其损失函数

$$\mathcal{L}(\omega,b) = \frac{{||\omega||}_{2}^{2}}{2} + \frac{C}{n}\sum_{i = 1}^{N}{max\{ 0,1 - y_{i}\left( \omega^{T}x_{i} + b \right)\}}$$

在$1 - y_{i}\left( \omega^{T}x_{i} + b \right) = 0$的位置不可导。我们希望找到一个合适的损失函数，使其在各个位置都可导，从而方便计算。

逻辑回归构造sigmoid函数，如下：  
$$sigmoid(z) = \frac{1}{1 + e^{- z}}$$

其中，  
$$sigmoid(0) = \frac{1}{2}$$

$$\lim_{z \rightarrow + \infty}{sigmoid(z)} = 1$$

$$\lim_{z \rightarrow - \infty}{sigmoid(z)} = 0$$

在SVM中，我们的目标是让标签为+1（正类）的样本的预测值为正值，标签为--1（负类）的样本的预测值为负值。在逻辑回归中，我们进一步将预测值输入进sigmoid函数中，得到的结果为样本类别为正类的概率。此外，为方便定义损失函数，在逻辑回归中，我们定义正类的标签仍为1，但负类的标签为0。我们的训练目标变为：让标签为1的样本被预测为正类的概率尽可能大；让标签为0的样本被预测为正类的概率尽可能小，即被预测为负类的概率尽可能大。由此，我们构造如下损失函数：  
$$L(\omega,b) = \frac{1}{N}\sum_{i = 1}^{N}\left( - y_{i}\log\frac{1}{1 + e^{- \left( \omega^{T}x_{i} + b \right)}} - \left( 1 - y_{i} \right)\log\left( 1 - \frac{1}{1 + e^{- \left( \omega^{T}x_{i} + b \right)}} \right) \right)$$

$$= \frac{1}{N}\sum_{i = 1}^{N}\left( - y_{i}\log\frac{1}{1 + e^{- \left( \omega^{T}x_{i} + b \right)}} - \left( 1 - y_{i} \right)\log\frac{1}{1 + e^{\omega^{T}x_{i} + b}} \right)$$

$$= \frac{1}{N}\sum_{i = 1}^{N}\left( y_{i}log\left( 1 + e^{- \left( \omega^{T}x_{i} + b \right)} \right) + \left( 1 - y_{i} \right)log\left( 1 + e^{\omega^{T}x_{i} + b} \right) \right)$$

令$\omega \leftarrow \left\lbrack \omega^{T},b \right\rbrack^{T},x_{i} \leftarrow \lbrack x_{i}^{T},1\rbrack$，则损失函数变为：

$$\mathcal{L}(\omega) = \frac{1}{N}\sum_{i = 1}^{N}\left( y_{i}log\left( 1 + e^{- x_{i}\omega} \right) + \left( 1 - y_{i} \right)log\left( 1 + e^{x_{i}\omega} \right) \right)$$

下面求解梯度：

$$\frac{\partial\mathcal{L}}{\partial\omega} = \frac{1}{N}\sum_{i = 1}^{N}{\left( \frac{1}{1 + e^{- x_{i}\omega}} - y_{i} \right)x_{i}}$$

然后利用梯度下降更新$\omega$即可。

## 2、多分类逻辑回归算法

多分类数据集$\mathcal{D =}\left\{ x_{i},y_{i} \right\}_{i = 1}^{N}$，其中$x_{i} \in \mathbb{R}^{d \times 1}$，$y_{i} \in \{ 1,2,\ldots,C\}$，共$C$个类别。

令$x_{i} \leftarrow \left\lbrack x_{i}^{T},1 \right\rbrack,i \in \{ 1,2,\ldots,N\}$。

在多分类问题中，模型参数$\omega \in \mathbb{R}^{(d + 1) \times C}$，共$C$个列，每列对应一个类别的通用特征。将$x_{i}$与每个类别的通用特征求相似度，即内积，相似度最高的值对应的类别即为$x_{i}$的预测类别。即：

$$z_{i} = x_{i}\omega,z_{i} \in \mathbb{R}^{1 \times C}$$

$z_{ij}$即为样本$x_{i}$与第$j$类的通用特征的相似度值。

定义softmax函数：

$${\widehat{y}}_{i} = softmax\left( z_{i} \right) = \left\lbrack \frac{e^{z_{it}}}{\sum_{j = 1}^{C}e^{z_{ij}}} \right\rbrack_{t = 1}^{C}$$

${\widehat{y}}_{ij}$即为样本$x_{i}$被预测为第$j$个类别的概率。

我们希望样本$x_{i}$被预测为第$y_{i}$类的概率最大，即：

$$\max_{\omega}{\widehat{y}}_{iy_{i}} = \max_{\omega}\frac{e^{z_{iy_{i}}}}{\sum_{j = 1}^{C}e^{z_{ij}}}$$

因此损失函数定义为：

$$\mathcal{L}(\omega) = \sum_{i = 1}^{N}\left( - log\frac{e^{z_{iy_{i}}}}{\sum_{j = 1}^{C}e^{z_{ij}}} \right) = - \sum_{i = 1}^{N}{log\frac{e^{z_{iy_{i}}}}{\sum_{j = 1}^{C}e^{z_{ij}}}}$$