## （一）人脸识别任务

人脸识别任务是从一个图像中识别所有包含人脸的矩形区域，并精准检测人脸区域中的关键点（如双眼、鼻尖、嘴角等）。

一种看似可行的方案是，将图像中包含人脸的矩形区域的左上角和右下角的坐标，以及关键点的坐标作为图像标签，通过损失函数将模型的输出与标签对齐。然而，这种方案存在的问题是，一张图片中所包含的人脸数量是不固定的，可能只包含一个，也可能包含非常多个。多任务级联卷积神经网络（MTCNN）能够较好地解决这一问题。

## （二）MTCNN的人脸识别过程

MTCNN由三级网络构成：

### （1）P-Net

P-Net用于候选框生成，即在一张图像中识别出所有可能包含人脸的矩形区域，其作用是快速过滤掉不是人脸的区域。

P-Net的输入是一张3 \* H \* W的RGB图像。输出是一张h \* w的热力图和一个4 \* h \* w的包含候选框位置信息的张量。其中h \< H, w \< W。

想象一个12 \* 12的滑动窗口从图像左上角开始滑动，其横向和纵向的滑动步长均为2。其所经过的图像中的每个12 \* 12的矩形区域，对应热力图中的一个数值，这个数值的取值位于(0, 1)区间，表示这个矩形区域中包含人脸的概率，即置信度值。这个滑动窗口经过的图像中的每个12 \* 12的矩形区域，同样对应P-Net输出的4 \* h \* w的张量中的一个4维向量 {Δx1, Δy1, Δx2, Δy2}。

对于图像中某个12 \* 12的矩形区域A，Δx1和Δy1分别为A所包含的人脸矩形框的左上角的横坐标和纵坐标相对于A的左上角的偏移量，Δx2和Δy2分别为A所包含的人脸矩形框的右下角的横坐标和纵坐标相对于A的左上角的偏移量。具体来说，其计算公式如下：

设A左上角在原图中的坐标为(x0, y0)；A所包含的人脸矩形框的左上角在原图中的坐标为(x1, y1)，右下角在原图中的坐标为(x2, y2)。则：

Δx1 = (x1 – x0) / 12  
Δy1 = (y1 – y0) / 12  
Δx2 = (x2 – x0) / 12  
Δy2 = (y2 – y0) / 12

当热力图中对应的置信度值高于某个阈值时，我们认为这个12 \* 12的矩形区域中包含人脸，只有这种情况下，这个4维向量的值才有意义。我们可以利用上述公式，根据P-Net生成的坐标偏移量推出候选人脸矩形框在原图中的位置坐标。这样，我们就得到了一个个候选的人脸矩形区域。

但一张图像的人脸区域可能很大（如自拍图像），而P-Net生成的热力图却是图像中的每个12 \* 12的很小的矩形区域中的人脸置信度。对于人脸区域较大的图像而言，每个12 \*12的矩形区域可能只包含人脸中的某个局部区域（如脸颊、鼻子等），导致模型无法通过局部区域识别出人脸的存在。因此，我们需要对数据进行如下处理：

将原图Img按照比例k缩放(0 \< k \< 1)，即将Img缩小为原来的k倍，得到Img1；再将Img1缩小为其k倍，得到Img2；将Img2缩小为其k倍，得到Img3……

将原图Img和缩小后的所有图像均输入到P-Net中，P-Net会为每张图像都生成若干候选矩形框。对于缩小后的图像，将其中的候选矩形框按照缩放比例对应到原图中，就得到了其在原图中的位置坐标。这样，P-Net就可以识别原图中不同大小的人脸区域。

### （2）R-Net

R-Net用于候选框筛选，即针对P-Net生成的候选矩形框，进一步确定其是否是人脸区域，并针对是人脸的区域进一步校准矩形框的位置和大小。

R-Net的输入是3 \* 24 \* 24的RGB图像，输出是这个图像的人脸置信度值和候选人脸矩形框的坐标偏移量（同样是上述4维向量）。

R-Net的输入图像来自P-Net为原始图像生成的若干候选矩形区域，将这些候选矩形区域缩放至24 \* 24大小，输入进R-Net中。如果置信度低于某个阈值，我们认为P-Net判断有误，即这个区域中不包含人脸；如果置信度高于这个阈值，我们认为这个区域中包含人脸，根据P-Net生成的候选框在原图中的位置坐标和R-Net生成的坐标偏移量，就可以得到R-Net生成的候选框在原图中的位置坐标。

### （3）O-Net

O-Net用于精细检测与关键点定位，即针对R-Net生成的候选框，进一步确定其是否是人脸区域，并针对是人脸的区域进一步校准矩形框的位置和大小，以及定位人脸区域中双眼、鼻尖和两个嘴角等5个关键点的位置坐标。

O-Net的输入是3 \* 48 \* 48的RGB图像，输出是这个图像中的人脸置信度值、候选人脸矩形框的坐标偏移量（同样是上述4维向量）、以及5个关键点的坐标偏移量（一个10维向量，每两个维度的值对应一个关键点的横纵坐标偏移量）。

O-Net的输入图像来自R-Net生成的若干候选矩形框，将这些候选矩形区域缩放至48 \* 48大小，输入进O-Net中。如果O-Net生成的置信度低于某个阈值，我们认为R-Net判断有误，即这个区域中不包含人脸；如果置信度高于这个阈值，我们认为这个区域中包含人脸，然后根据R-Net生成的候选框在原图中的位置坐标和O-Net生成的坐标偏移量，就可以得到O-Net生成的候选框在原图中的位置坐标和5个关键点在原图中的位置坐标。

## （三）MTCNN的训练

### （1）P-Net

P-Net的训练使用的数据集中，每张图像的标签为若干个4维向量，每个4维向量为其中一个人脸区域左上角和右下角的位置坐标（注意是整数类型的绝对位置坐标，而不是浮点数偏移量）。

设原图的宽度和高度分别为W和H。

首先，从原图中采样若干负样本。具体做法是，每次从原图中随机采样一个边长位于\[12, min{W / 2, H / 2}\]的正方形区域。将这个正方形区域和图像标签所标注的原图中的每个人脸区域求交并比（IoU，即两个区域相交部分的面积与两个区域并集的面积的比值），取最大的交并比，若其值小于阈值a，则选择其作为负样本，标签为0。然后将其缩放至12 \* 12。

然后对于原图中的每个人脸区域，在这个区域附近，采样几个困难负样本、几个正样本和几个部分人脸样本。

困难负样本即位于人脸区域附近、但与人脸区域最大交并比小于阈值a的区域。这些区域容易被模型误判为正样本，将其加入训练数据中可以增强模型的健壮性。设人脸区域左上角坐标为(x1, y1)，右下角坐标为(x2, y2)。采样的正方形区域边长同样位于\[12, min{W / 2, H / 2}\]。设边长为size，则采样区域的左上角横坐标位于\[x1 – size, x2 + 1\]，采样区域的左上角的纵坐标位于\[y1 – size, y2 + 1\]。也就是在人脸区域及其四周随机采样。困难负样本的标签为0。采样区域同样需要被缩放至12 \* 12。

正样本即位于人脸区域附近、且与人脸区域最大交并比大于阈值b的区域。设人脸区域中心点的坐标为(x0, y0)，为增强训练数据的多样性，应允许样本的中心位置略微偏离(x0, y0)。因此样本中心点的横坐标为\[x0 – 0.2 \* w, x0 + 0.2 \* w\]上的一个随机值，样本中心点的纵坐标为\[y0 – 0.2 \* h, y0 + 0.2 \* h\]上的一个随机值。样本的边长同样允许一定幅度的偏差，因此样本边长为\[min{0.8 \* w, 0.8 \* h}, max{1.25 \* w, 1.25 \* h}\]上的一个随机值。正样本同样需要缩放到12 \* 12大小。正样本标签为1，除此之外，还要计算出人脸区域左上角和右下角的坐标偏移量。

部分人脸样本为包含部分人脸、但非全部人脸的正方形区域。采样方法与正样本相同，但最大交并比的值位于阈值a和b之间。部分人脸样本的标签值为–1。除此之外，还要计算出人脸区域左上角和右下角的坐标偏移量。由于P-Net不仅需要识别一个区域是否包含人脸，还要识别人脸候选框的位置。部分人脸区域虽然不能帮助模型训练其识别是否为人脸的能力，但可以帮助模型提升定位人脸位置的精确性。

P-Net训练的损失函数分为两个部分，其一为用于分类的交叉熵损失（将人脸置信度对其标签），其二为用于回归的均方误差（将模型生成的人脸区域坐标偏移量对齐标签）。正样本和负样本用于分类，正样本和部分人脸样本用于回归。

### （2）R-Net

R-Net使用的训练数据集与P-Net相同。首先，利用（二）中所述的方法将原始图像输入进P-Net中，生成若干人脸候选框。对于每个人脸候选框，求其与真实人脸区域的最大交并比。如果最大交并比小于阈值a，则作为负样本，标签为0；位于阈值a和阈值b之间，则作为部分样本，标签为–1，并求出人脸区域的坐标偏移量作为回归标签；大于阈值b则作为正样本，标签为1，并求出人脸区域的坐标偏移量作为回归标签。每个样本被缩放至24 \* 24大小。这些样本作为R-Net的训练数据。

R-Net训练的损失函数与P-Net相同。

### （3）O-Net

O-Net使用另一个数据集。每张图像的标签为若干个14维向量，对应人脸区域左上角和右下角的绝对位置坐标和5个关键点的绝对位置坐标。

首先按照（二）中所述的方法，先将原始图像输入进P-Net，生成若干候选框。再将P-Net生成的候选框输入进R-Net，生成若干校准后的候选框。取每个R-Net生成的候选框，按照与（2）中类似的方式生成O-Net的训练数据。注意对于正样本和部分人脸样本，除了计算人脸区域左上角和右下角的坐标偏移量外，还需计算5个关键点的坐标偏移量。每个样本均缩放至48 \* 48大小。

正样本和负样本用于分类，正样本和部分人脸样本用于回归。

---

```python
class PNet(nn.Module):
    ''' PNet '''
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1), # conv1
            nn.PReLU(), # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2), # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1), # conv2
            nn.PReLU(), # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # conv3
            nn.PReLU() # PReLU3
        )

        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

        # weight initiation with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return label, offset

class RNet(nn.Module):
    ''' RNet '''
    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1), # conv1
            nn.PReLU(), # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2), # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1), # conv2
            nn.PReLU(), # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2), # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1), # conv3
            nn.PReLU() # prelu3
        )
        self.fc = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU() # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)

        # weight initiation with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.prelu4(x)
        # detection
        det = F.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        return det, box

class ONet(nn.Module):
    ''' ONet '''
    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1), # conv1
            nn.PReLU(), # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2), # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # conv2
            nn.PReLU(), # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2), # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )
        self.fc = nn.Linear(128*2*2, 256)
        self.prelu5 = nn.PReLU() # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)

        # weight initiation with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.prelu5(x)
        # detection
        det = F.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        return det, box, landmark