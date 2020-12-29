# -*- coding:utf-8 -*-
# @Time : 2020/12/29 8:58 下午
# @Author : huichuan LI
# @File : client.py
# @Software: PyCharm
from sklearn.datasets import make_circles

import sys

sys.path.append('../')

import grpc

# 获取同心圆状分布的数据，X的每行包含两个特征，y是1/0类别标签
X, y = make_circles(200, noise=0.1, factor=0.2)
y = y  # 将标签转化为1/-1
import HuichuanFlow_serving as mss
from HuichuanFlow_serving.serving import serving_pb2_grpc

from HuichuanFlow_serving.serving.proto import serving_pb2
import numpy as np


class MatrixSlowServingClient(object):
    def __init__(self, host):
        self.stub = serving_pb2_grpc.HuichuanFlowServingStub(
            grpc.insecure_channel(host))

        print('[GRPC] Connected to MatrixSlow serving: {}'.format(host))

    def Predict(self, mat_data_list):
        req = serving_pb2.PredictReq()
        for mat in mat_data_list:
            proto_mat = req.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))
        resp = self.stub.Predict(req)
        print(resp)
        return resp


host = '127.0.0.1:5000'
client = MatrixSlowServingClient(host)

for index in range(len(X)):
    img = X[index]
    label = y[index]
    resp = client.Predict([img])
    resp_mat_list = []
    for proto_mat in resp.data:
        dim = tuple(proto_mat.dim)
        mat = np.mat(proto_mat.value, dtype=np.float32)
        mat = np.reshape(mat, dim)
        resp_mat_list.append(mat)
    pred = np.argmax(resp_mat_list[0])
    gt = label
    print('model predict {} and ground truth: {}'.format(
        np.argmax(pred), gt))
