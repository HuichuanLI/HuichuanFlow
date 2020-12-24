# -*- coding:utf-8 -*-
# @Time : 2020/12/24 1:35 下午
# @Author : huichuan LI
# @File : GRU.py
# @Software: PyCharm
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


# 构造正弦波和方波两类样本的函数
def get_sequence_data(dimension=10, length=10,
                      number_of_examples=1000, train_set_ratio=0.7, seed=42):
    """
    生成两类序列数据。
    """
    xx = []

    # 正弦波
    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))

    # 方波
    xx.append(np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        x = xx[i]
        for j in range(number_of_examples // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension))  # 加入噪声
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    # 把各个类别的样本合在一起
    data = np.concatenate(data, axis=0)
    print(data[0].shape)

    # 随机打乱样本顺序
    np.random.shuffle(data)

    # 计算训练样本数量
    train_set_size = int(number_of_examples * train_set_ratio)  # 训练集样本数量
    plt.plot(np.arange(0, len(data[1, :-2])), data[1, :-2])

    # 将训练集和测试集、特征和标签分开
    return (data[:train_set_size, :-2].reshape(-1, length, dimension),
            data[:train_set_size, -2:],
            data[train_set_size:, :-2].reshape(-1, length, dimension),
            data[train_set_size:, -2:])


# 构造RNN
seq_len = 10  # 序列长度
dimension = 10  # 输入维度
status_dimension = 2  # 状态维度

signal_train, label_train, signal_test, label_test = get_sequence_data(length=seq_len, dimension=dimension)

import sys

sys.path.append('../')
import HuichuanFlow as hf

# 输入向量节点
inputs = [hf.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]

# 输入门
# 输入权值矩阵
U_z = hf.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_z = hf.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_z = hf.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 输出门
# 输入权值矩阵
U_r = hf.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_r = hf.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_r = hf.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

U_x = hf.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)
W_x = hf.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

last_step = None  # 上一步的输出，第一步没有上一步，先将其置为 None
for iv in inputs:
    z = hf.ops.Add(hf.ops.MatMul(U_z, iv), b_z)
    r = hf.ops.Add(hf.ops.MatMul(U_r, iv), b_r)
    h_t = hf.ops.MatMul(U_x, iv)

    if last_step is not None:
        z = hf.ops.Add(hf.ops.MatMul(W_z, last_step), z)
        r = hf.ops.Add(hf.ops.MatMul(W_r, last_step), r)
    z = hf.ops.Logistic(z)
    r = hf.ops.Logistic(r)
    if last_step is not None:
        h_t = hf.ops.Add(hf.ops.MatMul(W_x, hf.ops.Multiply(last_step, r)), h_t)
    h_t = hf.ops.ReLU(h_t)
    h = hf.ops.Multiply(z, h_t)
    last_step = h

fc1 = hf.layer.fc(last_step, status_dimension, 12, "ReLU")  # 第一全连接层
# fc2 = hf.layer.fc(fc1, 40, 10, "ReLU")  # 第二全连接层
output = hf.layer.fc(fc1, 12, 2, "None")  # 输出层

# 概率
predict = hf.ops.Logistic(output)

# 训练标签
label = hf.core.Variable((2, 1), trainable=False)

# 交叉熵损失
loss = hf.ops.CrossEntropyWithSoftMax(output, label)

# 训练
learning_rate = 0.005
optimizer = hf.optimizer.Adam(hf.default_graph, loss, learning_rate)

batch_size = 1

for epoch in range(50):

    batch_count = 0
    for i, s in enumerate(signal_train):

        # 将每个样本各时刻的向量赋给相应变量
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        label.set_value(np.mat(label_train[i, :]).T)

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0

    pred = []
    for i, s in enumerate(signal_test):

        # 将每个样本各时刻的向量赋给相应变量
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)

    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))
