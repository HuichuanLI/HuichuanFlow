# -*- coding:utf-8 -*-
# @Time : 2020/12/21 10:32 下午
# @Author : huichuan LI
# @File : lr_mnist.py
# @Software: PyCharm

import sys

sys.path.append('../..')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import HuichuanFlow as ms

from sklearn.datasets import load_digits

mnist = load_digits()

# 加载MNIST数据集，只取5000个样本
X, y = mnist.data, mnist.target

X = X.reshape(-1, 64)
X, y = X[:5000] / 255, y.astype(np.int)[:5000]
# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

# 构造计算图：输入向量，是一个784x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(64, 1), init=False, trainable=False)

# One-Hot类别标签，是10x1矩阵
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

# 输出层，10个神经元，无激活函数
output = ms.layer.fc(x, 64, 10, None)

# 概率输出
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.001

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小为64
batch_size = 64

# 训练执行30个epoch
for epoch in range(30):

    # 批计数器清零
    batch_count = 0

    # 遍历训练集中的样本
    for i in range(len(X)):

        # 取第i个样本，构造784x1矩阵对象
        feature = np.mat(X[i]).T

        # 取第i个样本的One-Hot标签，10x1矩阵
        label = np.mat(one_hot_label[i]).T

        # 将特征赋给x节点，将标签赋给one_hot节点
        x.set_value(feature)
        one_hot.set_value(label)

        # 调用优化器的one_step方法，执行一次前向传播和反向传播
        optimizer.one_step()

        # 批计数器加1
        batch_count += 1

        # 若批计数器大于等于批大小，则执行一次更新，并清零计数器
        if batch_count >= batch_size:
            # 打印当前epoch数，迭代数与损失值
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(
                epoch + 1, i + 1, loss.value[0, 0]))

            # 优化器执行更新
            optimizer.update()
            batch_count = 0

    # 每个epoch结束后评估模型的正确率
    pred = []

    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(X)):
        feature = np.mat(X[i]).T
        x.set_value(feature)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value.A.ravel())  # 模型的预测结果：10个概率值

    pred = np.array(pred).argmax(axis=1)  # 取最大概率对应的类别为预测类别

    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (y == pred).astype(np.int).sum() / len(X)

    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
