# -*- coding:utf-8 -*-
# @Time : 2020/12/29 7:20 下午
# @Author : huichuan LI
# @File : serving.py
# @Software: PyCharm
import sys

sys.path.append('../')

import HuichuanFlow_serving as hfs

print(hfs.serving.HuichuanFlowServingService)

serving = hfs.serving.HuichuanFLowServer(
    host='127.0.0.1:5000', root_dir='./epoches20', model_file_name='my_model.json', weights_file_name='my_weights.npz')

serving.serve()
