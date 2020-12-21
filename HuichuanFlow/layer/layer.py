# -*- coding:utf-8 -*-
# @Time : 2020/12/21 10:24 下午
# @Author : huichuan LI
# @File : layer.py
# @Software: PyCharm
from ..core import *
from ..ops import *


def fc(input, input_size, size, activation):
    """
    :param input: 输入向量
    :param input_size: 输入向量的维度
    :param size: 神经元个数，即输出个数（输出向量的维度）
    :param activation: 激活函数类型
    :return: 输出向量
    """
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
