# -*- coding:utf-8 -*-
# @Time : 2020/12/31 9:55 下午
# @Author : huichuan LI
# @File : CNN.py
# @Software: PyCharm
import sys

sys.path.append('../')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import HuichuanFlow as ms

# 加载MNIST数据集，只取5000个样本
from sklearn.datasets import load_digits

mnist = load_digits()

# 加载MNIST数据集，只取5000个样本
X, y = mnist.data, mnist.target

X, y = X[:5000] / 255, y.astype(np.int)[:5000]

# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

# 输入图像尺寸
img_shape = (8, 8)

# 输入图像
x = ms.core.Variable(img_shape, init=False, trainable=False)

# One-Hot标签
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

# 第一卷积层
conv1 = ms.layer.conv([x], img_shape, 3, (2, 2), "ReLU")

# 第一池化层
pooling1 = ms.layer.pooling(conv1, (2, 2), (2, 2))

# # 第二卷积层
# conv2 = ms.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

# # 第二池化层
# pooling2 = ms.layer.pooling(conv2, (3, 3), (2, 2))


# 全连接层
fc1 = ms.layer.fc(ms.ops.Concat(*pooling1), 16 * 3, 20, "ReLU")

# 输出层
output = ms.layer.fc(fc1, 20, 10, "None")

# 分类概率
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.005

# 优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小
batch_size = 32

# 训练
for epoch in range(60):

    batch_count = 0

    for i in range(len(X)):

        feature = np.mat(X[i]).reshape(img_shape)
        label = np.mat(one_hot_label[i]).T

        x.set_value(feature)
        one_hot.set_value(label)

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    accuracy = (y == pred).astype(np.int).sum() / len(X)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
