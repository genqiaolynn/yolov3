# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2020/12/12 17:52'


class Config:
    def __init__(self):
        iou = {0: 'default', 1: 'GIoU', 2: 'DIoU', 3: 'CIoU'}
        self.IOU = iou[1]

cfg = Config()