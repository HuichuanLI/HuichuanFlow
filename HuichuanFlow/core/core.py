# -*- coding:utf-8 -*-
# @Time : 2020/12/28 10:22 下午
# @Author : huichuan LI
# @File : core.py
# @Software: PyCharm
from .node import Variable
from .graph import default_graph


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None
