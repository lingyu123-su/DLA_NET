## 工程整体介绍

### 工程文件介绍
#### dataset文件夹
* 包含数据集及一个处理数据集的文件
* readMat.py的作用是把photo对应的sketch改名为：photoID_repeatID，其中photo为sketch对应的类别，repeatID是该类别下重复出现的次数，例如：1_1.png，表示对应标签1的第1张草图。

#### options文件夹
* 包含train_options.py文件
* 此文件定义了一些训练时用到的参数，包括训练集路径、测试集路径、模型保存路径、训练时的batch_size、学习率等。

#### results文件夹
* 包含checkpoint文件夹和log文件夹
* checkpoint文件夹用于保存训练好的模型，log文件夹用于保存训练过程中的日志。

#### test文件夹
* 包含一些辅助理解的文件和用来测试语法，无主要内容

#### utils文件夹
* 包含2个文件ckpt_utils.py，Logger.py
* ckpt_utils.py用于保存模型，定义了一些保存模型时的细节。
* Logger.py用于保存训练过程中的日志，定义了一些保存日志时的细节。

#### dataset.py
* 此文件定义了训练数据和测试图片，测试草图的类，包括数据的读取、预处理等
* 训练数据返回的是pair对，测试图片返回的是图片和标签，测试草图返回的是草图和标签。（1_1）

#### LA.py
* 此文件定义了LA距离的计算方法
* LA的算法相对简单，只需要算出图片特征图和草图特征图对应位的欧式距离，然后用2norm求和，得到最终的距离。
* 输入是草图和图片的特征图，返回的是一个（batch_size, batch_size）的矩阵，矩阵中的每个元素表示两个特征图之间的距离。

#### DLA.py
* 此文件定义了DLA距离的计算方法
* DLA的距离类似于卷积，每个特征和整个特征图上的每个特征算欧氏距离，然后取最小值作为当前特征到整个特征图的距离，算出每个特征到整个特征图的距离后，用2norm求和，得到最终的距离。
* 输入是草图和图片的特征图，返回的是一个（batch_size, batch_size）的矩阵，矩阵中的每个元素表示两个特征图之间的距离。  

#### network.py
* 此文件定义了网络结构
* 网络结构相对简单，图片分支和草图分支的模型时一样的，不用分别定义
* backbone是resnet50，当提取中间特征图时，使用-3，也就是去掉后面的conv4，gap和fc层，只保留前面的conv1-conv3层，这样得到的特征图大小为（batch_size, 1024, 16, 16）
* 当提取高层特征时，使用-2，也就是去掉后面的gap，fc层，只保留前面的conv1-conv4层，这样得到的特征图大小为（batch_size, 2048, 8, 8）
* 当提取最后一层特征时，使用-1，也就是去掉后面的fc层，输出的特征图大小为（batch_size, 2048, 1, 1）

#### loss.py
* 此文件定义了损失函数，论文中为batch all triplet loss
* 之前计算的LA和DLA都是矩阵，矩阵中每个值都是两个特征图之间的距离，然后根据triplet loss的公式，选出正负样本，再进行计算。
* 输入是dis矩阵和margin，输出是loss

#### triplet_dataset.py
* 此文件定义了训练数据集,返回正负样本对和一张草图
* 从训练集中选取一张图片和对应的草图，再选取一张不同的作为负样本，返回pos——photo，sketch，neg——photo

#### triplet_train.py
* 此文件定义了训练过程，采用了传统的三个样本的方法，因为图片共享参数，所以只需要两个branch，获得正负样本的距离，按照triplet loss的公式计算loss，然后反向传播，更新参数
* 出了点问题，准确率很低，成功复现看train.py

#### train.py
* 此文件定义了训练过程
* 根据传参确定超参数和模型，然后开始训练，用tqdm显示训练进度
* 使用triplet loss训练，每个epoch都会计算一次验证集的准确率，保存最佳模型

#### test.py
* 此文件定义了测试过程
* 这个文件没有用opt，根据要测试的模型，修改文件了match的方式和模型的路径，返回的是准确率

#### eval.py
* 此文件定义了评估过程
* 包括一个loss相关的类，用来算loss，一个acc相关的类，用来算准确率

### 项目难点分析
#### 1. LocalL2
本来以为是加了一个计算，其实只是多给调的函数里传了个参数
#### 2. LA
LA的原理很清晰，比较困难的是实现，核心函数是torch.cdist，只要把传进来的特征图改成要求的形状，再丢进去就好了。
#### 3. DLA
核心还是cdist函数，难点依然是变换输入数据
#### 4. Triplet Loss
使用了论文中提到的batch all triplet loss，把一个batch中的所有样本都用上了

#### 5. eval
计算草图和所有图片的距离，然后选出最小的，如果最小的距离对应的图片的标签和草图的标签一样，就算对了，否则算错了

### 项目复现结果
* 固定住bn后收敛更快了
#### LA
* 使用中间特征，LocalL2，batch_size=32，lr=0.0001，epoch=100，margin=0.1，准确率为0.43左右，略高于论文
* 使用高层特征，LocalL2，batch_size=32，lr=0.0001，epoch=100，margin=0.1，准确率为0.42左右，略高于论文

#### DLA
* 使用中间特征，LocalL2，batch_size=32，lr=0.0001，epoch=100，margin=0.1，准确率为0.5左右，略低于论文
* 使用高层特征，LocalL2，batch_size=32，lr=0.0001，epoch=100，margin=0.1，准确率为0.43左右，略高于论文
* 使用中层特征，无LocalL2，batch_size=32，lr=0.0001，epoch=100，margin=0.1，准确率为0.47左右，略低于论文
* 上面的准确率可能有1%左右的浮动，总体上来说和论文里差距不是很大

#### 测试流程
* 先把数据集放进去，然后执行readMat.py，把草图改名
* 以中层特征，LocalL2的DLA为例（就是文中的DLA_NET）
* 在network里修改init方法，使用middle feature map，修改forword，使用LocalL2
* 在test.py里面修改match方法为DLA的match方法，修改模型路径为DLA的模型路径（./results\checkpoint\DLA_LocalL2_mid_frozeBN\ckpt-best.pth）
* 运行test.py，得到准确率为0.5
