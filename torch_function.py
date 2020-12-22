# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2020/12/22 15:15'


import torch


x = torch.tensor([1, 2, 3])
print(x.repeat(4))   # 起到的作用是将前面看成一个整体，复制4遍
print('-------------------------')
print(x.repeat(4, 1))  # 起到的作用是将前面看成一个整体，复制4遍
print('------------')
print(x.repeat(4, 2))

# tensor.repeat(*sizes)   参数*sizes指定了原始张量在各维度上复制的次数。整个原始张量作为一个整体进行复制。