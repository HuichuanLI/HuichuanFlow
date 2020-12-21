# -*- coding:utf-8 -*-
# @Time : 2020/12/20 9:59 下午
# @Author : huichuan LI
# @File : loss.py
# @Software: PyCharm


import numpy as np

from ..core import node


class LossFunction(node.Node):
    '''
    定义损失函数抽象类
    '''
    pass


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class LogLoss(LossFunction):
    """
    对数损失函数
    """
    def compute(self):
        assert len(self.parents) == 1

        x = self.parents[0].value

        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))

        return np.diag(diag.ravel())
