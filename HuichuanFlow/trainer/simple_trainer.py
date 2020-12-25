# -*- coding:utf-8 -*-
# @Time : 2020/12/25 12:44 下午
# @Author : huichuan LI
# @File : simple_trainer.py
# @Software: PyCharm
from .trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        '''
        不做统一的初始化操作，使用节点自身的初始化方法
        '''
        pass

    def _optimizer_update(self):
        self.optimizer.update()
