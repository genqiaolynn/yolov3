# -*- coding:utf-8 -*-
from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config.config import cfg
import sys
sys.float_info.dig   # 可看当前机器支持的浮点位数   15


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)     # TODO get_iou
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    '''
    :param wh1:  聚类得到先验框的宽高
    :param wh2:  target box
    :return: 1和所有的2的IOU值
    '''
    # wh1: anchor (2) // anchors: (3, 2)
    # wh2: target_wh (n_boxes, 2)
    # wh2.t(): (2, n_boxes)
    # e.g
    # wh1 = torch.Tensor(np.random.random((2)))
    # wh2 = torch.Tensor(np.random.random((2, 5)))
    # wh1: tensor([0.5401, 0.1042])
    # wh2: tensor([[0.6327, 0.2902, 0.6464, 0.3725, 0.8088],
    #              [0.2016, 0.0760, 0.9205, 0.3213, 0.7930]])
    # w1: tensor(0.5401)
    # h1: tensor(0.1042)
    # w2: tensor([0.6327, 0.2902, 0.6464, 0.3725, 0.8088])
    # h2: tensor([0.2016, 0.0760, 0.9205, 0.3213, 0.7930])
    # torch.min()
    # minw = torch.min(w1, w2)——>tensor([0.5401, 0.2902, 0.5401, 0.3725, 0.5401])
    # minh = torch.min(h1, h2)——>tensor([0.1042, 0.0760, 0.1042, 0.1042, 0.1042])
    # inter_area = minw * minh = tensor([0.0563, 0.0221, 0.0563, 0.0388, 0.0563])
    # stack后final res shape:
    # e.g.(3, 5) ——>(num_anchor, n_boxes)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    # 16 <--这里表明你系统提供的float有效位为16位
    # a = 1e-16
    # 1.0 + a
    # 1.0
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bboxes_iou_(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious


def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious


def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious, min=0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


def bbox_overlaps_giou(bboxes11, bboxes22):
    bboxes1 = torch.cat((torch.cat((bboxes11[:, 0] - bboxes11[:, 2] / 2, bboxes11[:, 1] - bboxes11[:, 3] / 2), 1),
                         torch.cat((bboxes11[:, 0] + bboxes11[:, 2] / 2, bboxes11[:, 1] + bboxes11[:, 3] / 2), 1)), 1)

    bboxes2 = torch.cat((torch.cat((bboxes22[:, 0] - bboxes22[:, 2] / 2, bboxes22[:, 1] - bboxes22[:, 3] / 2), 1),
                         torch.cat((bboxes22[:, 0] + bboxes22[:, 2] / 2, bboxes22[:, 1] + bboxes22[:, 3] / 2), 1)), 1)

    # bboxes1 = torch.from_numpy(np.array([bboxes11[:, 0] - bboxes11[:, 2] / 2, bboxes11[:, 1] - bboxes11[:, 3] / 2,
    #            bboxes11[:, 0] + bboxes11[:, 2] / 2, bboxes11[:, 1] + bboxes11[:, 3] / 2]))
    #
    # bboxes2 = torch.from_numpy(np.array([bboxes22[:, 0] - bboxes22[:, 2] / 2, bboxes22[:, 1] - bboxes22[:, 3] / 2,
    #            bboxes22[:, 0] + bboxes22[:, 2] / 2, bboxes22[:, 1] + bboxes22[:, 3] / 2]))

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


def bboxes_iou(bboxes1, bboxes2, IOU=cfg.IOU):
    if IOU == 'default':
        iou = bbox_overlaps_iou(bboxes1, bboxes2)
    elif IOU == 'GIoU':
         iou = bbox_overlaps_giou(bboxes1, bboxes2)
    elif IOU == 'DIoU':
        iou = bbox_overlaps_diou(bboxes1, bboxes2)
    elif IOU == 'CIoU':
        iou = bbox_overlaps_ciou(bboxes1, bboxes2)
    return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets_raw(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    generate masks & t·
    :param pred_boxes: 预测的bbox(0, 13) (b, num_anchor, grid_size, grid_size, 4) -> (b, 3, 13, 13, 4)
    :param pred_cls: 预测的类别概率(0, 1) (b, num_anchor, grid_size, grid_size, n_classes) -> (b, 3, 13, 13, 80)
    :param target: label(0, 1) (n_boxes, 6), 第二个维度有6个值，分别为: box所属图片在本batch中的index， 类别index， xc, yc, w, h
    :param anchors: tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) (num_anchor, 2) -> (3, 2-)->aw, ah
    :param ignore_thres: hard coded, 0.5
    :return: masks & t·
    """

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)     # batch size
    nA = pred_boxes.size(1)     # anchor size: 3
    nC = pred_cls.size(-1)      # class size: 80
    nG = pred_boxes.size(2)     # grid size: 13

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)      # (b, 3, 13, 13)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)    # (b, 3, 13, 13)    # mostly candidates are noobj
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)   # (b, 3, 13, 13)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)   # (b, 3, 13, 13)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)           # (b, 3, 13, 13)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)     # (b, 3, 13, 13, 80)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG      # (0, 1)->(0, 13) (n_boxes, 4)
    gxy = target_boxes[:, :2]               # (n_boxes, 2)
    gwh = target_boxes[:, 2:]               # (n_boxes, 2)
    # Get anchors with best iou
    # 仅依靠w&h 计算target box和anchor box的交并比， (num_anchor, n_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])     # TODO bbox_wh_iou
    # e.g, 每个anchor都和每个target bbox去算iou，结果存成矩阵(num_anchor, n_boxes)
    #                box0    box1    box2    box3    box4
    # ious=tensor([[0.7874, 0.9385, 0.5149, 0.0614, 0.3477],    anchor0
    #              [0.2096, 0.5534, 0.5883, 0.2005, 0.5787],    anchor1
    #              [0.8264, 0.6750, 0.4562, 0.2156, 0.7026]])   anchor2
    # best_ious:
    # tensor([0.8264, 0.9385, 0.5883, 0.2156, 0.7026])
    # best_n:
    # 属于第几个bbox：0, 1, 2, 3, 4
    #       tensor([2, 0, 1, 2, 2])   属于第几个anchor
    best_ious, best_n = ious.max(0)     # 最大iou, 与target box交并比最大的anchor的index // [n_boxes], [n_boxes]

    # Separate target values
    # target[:, :2]: (n_boxes, 2) -> img index, class index
    # target[].t(): (2, n_boxes) -> b: img index in batch, torch.Size([n_boxes]),
    #                               target_labels: class index, torch.Size([n_boxes])
    b, target_labels = target[:, :2].long().t()
    # gxy.t().shape = shape(gwh.t())=(2, n_boxes)
    gx, gy = gxy.t()            # gx = gxy.t()[0], gy = gxy.t()[1]
    gw, gh = gwh.t()            # gw = gwh.t()[0], gh = gwh.t()[1]
    gi, gj = gxy.long().t()     # .long()去除小数点
    # Set masks，这里的b是batch中的第几个
    obj_mask[b, best_n, gj, gi] = 1     # 物体中心点落在的那个cell中的，与target object iou最大的那个3个anchor中的那1个，被设成1
    noobj_mask[b, best_n, gj, gi] = 0   # 其相应的noobj_mask被设成0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # ious.t():
    # shape: (n_boxes, num_anchor)
    # i: box id
    # b[i]: img index in that batch
    # E.g 假设有4个boxes，其所属图片在batch中的index为[0, 0, 1, 2], 即前2个boxes都属于本batch中的第0张图
    #     则b[0] = b[1] = 0 都应所属图片在batch中的index，即batch中的第几张图
    for i, anchor_ious in enumerate(ious.t()):
        # 如果anchor_iou>ignore_thres，则即使它不是obj(非best_n)，同样不算noobj
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()     # x_offset (0, 1)
    ty[b, best_n, gj, gi] = gy - gy.floor()     # y_offset (0, 1)
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # 这两个是为了评价用，不参与实际回归
    # pred_cls与target_label匹配上了，则class_mask所对应的grid_xy为1，否则为0
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    # iou_scores[b, best_n, gj, gi] = bboxes_iou_(pred_boxes[b, best_n, gj, gi], target_boxes, xyxy=False, GIoU=cfg.IOU,
    #                                             DIoU=cfg.IOU, CIoU=cfg.IOU)  # raw

    tconf = obj_mask.float()        # target confidence
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    '''
    本函数的作用: generate masks & t.
    :param pred_boxes: 预测目标的四个值xywh shape -> [batch_size, 3, grid_size, grid_size, 4] --> (b, 3, 13, 13, 4)
    :param pred_cls: 预测目标的所有分类概率 shape -> [batch_size, 3, grid_size, grid_size, classes_num] --> (b, 3, 13, 13, 80)
    :param target:  真实目标的相关信息(图片索引,class_id,x,y,w,h) shape -> [len(target), 6]
    [len(target), 6] 第二个维度有6个值，分别为：box所在图片在本batch中的index，类别index，xc，yc，w，h   label(0, 1)
    :param anchors: anchors  某一YOLO层下的anchor尺寸 shape -> [3, 2]
    实例: tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]])  (num_anchor, 2) -> (3, 2)->aw, ah
    :param ignore_thres: iou忽略阈值,当iou超过这一值时,将noobj_mask设为 0
    TODO 这里的ignore_thres不是nms那种作用，这里是低于这个阈值的直接判0，高于这个值的判1，注意区分开！！！
    :param grid_size: 某一YOLO层下的grid尺寸
    :return: iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf 详情见下面注释
    :return: mask & t.
    '''
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)     # batch_size
    nA = pred_boxes.size(1)     # anchor_size:3   每个特征图里有几个先验框
    nC = pred_cls.size(-1)      # classes size
    nG = pred_boxes.size(2)     # grid cell size: 13

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)   # (b, 3, 13, 13)  most candidates are noobj  给出来的参数比值1:100
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)    # (b, 3, 13, 13, 80)   class_mask 后面加上了classes

    target = target[target.sum(dim=1) != 0]

    # Convert to position relative to box
    # 源target是padding后的0~1之间相对坐标,现在需要转换为以grid_size为单位下的坐标 (len(target),4) 方便下面计算box_iou
    target_boxes = target[:, 2:6] * nG      # (0, 1)->(0, 13) (n_boxes, 4)
    gxy = target_boxes[:, :2]               # (n_boxes, 2)
    gwh = target_boxes[:, 2:]               # (n_boxes, 2)
    # Get anchors with best iou
    # ious.shape -> (3,len(target)) 三种anchors尺寸下和各个target目标的宽高iou大小
    # 仅依靠w&h 计算target box和anchor box的交并比， (num_anchor, n_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # e.g, 每个anchor都和每个target bbox去算iou，结果存成矩阵(num_anchor, n_boxes)
    #                box0    box1    box2    box3    box4
    # ious=tensor([[0.7874, 0.9385, 0.5149, 0.0614, 0.3477],    anchor0
    #              [0.2096, 0.5534, 0.5883, 0.2005, 0.5787],    anchor1
    #              [0.8264, 0.6750, 0.4562, 0.2156, 0.7026]])   anchor2
    # best_ious:
    # tensor([0.8264, 0.9385, 0.5883, 0.2156, 0.7026])
    # best_n:
    # 属于第几个bbox：0, 1, 2, 3, 4
    # tensor([2, 0, 1, 2, 2])   属于第几个anchor
    # 获取每个真实box与3种anchors最大iou及anchor索引  (len(target),)
    best_ious, best_n = ious.max(0)    # 最大iou, 与target box交并比最大的anchor的index // [n_boxes], [n_boxes]
    # IOU比较这里是每张特征图上的值对比，是列对比，拿到最大的IOU对应的box及其索引值
    # Separate target values
    # 每个target所在图片在一个batch中的索引及目标种类id,注意这里的i_in_batch和target_labels可能会重复的,即一张图片中有两个同类目标！！！
    # target[:, :2]: (n_boxes, 2) -> img index, class index
    # target[].t(): (2, n_boxes) -> b: img index in batch, torch.Size([n_boxes]),
    # target_labels: class index, torch.Size([n_boxes])
    index_in_batch, target_labels = target[:, :2].long().t()
    n_gpus = len(cfg.n_gpu.split(','))
    img_cnt_per_gpu = int(cfg.batch_size/n_gpus)

    b = index_in_batch % img_cnt_per_gpu    # 将那几张图分到第一块GPU，哪几张图分到第二张GPU卡..
    # gxy.t().shape = shape(gwh.t())=(2, n_boxes)
    # gxy.t()是为了把shape从n x 2 变成 2 x n。
    # gi, gj = gxy.long().t()，是通过.long的方式去除小数点，保留整数。
    # 如此便可以设置masks。
    # b是指第几个target。
    # gi, gj 便是特征图中对应的左上角的坐标。
    gx, gy = gxy.t()          # gx = gxy.t()[0], gy = gxy.t()[1]
    gw, gh = gwh.t()          # gw = gwh.t()[0], gh = gwh.t()[1]
    gi, gj = gxy.long().t()   # .long()去除小数点

    # Set masks
    gi[gi < 0] = 0
    gj[gj < 0] = 0
    gi[gi > nG - 1] = nG - 1
    gj[gj > nG - 1] = nG - 1

    # 目标掩膜 有真实目标的位置为1,否则默认为0
    obj_mask[b, best_n, gj, gi] = 1
    # TODO obj_mask表示有物体落在特征图中某一个cell的索引，所以在初始化的时候置0，如果有物体落在那个cell中，那个对应的位置会置1
    noobj_mask[b, best_n, gj, gi] = 0     # 非目标掩膜 有真实目标的位置为0,否则默认为1,与obj_mask对立
    # TODO noobj表示没有物体落在特征图中某一个cell的索引,所以在初始化的时候置1，如果没有有物体落在那个cell中，那个对应的位置会置0
    # 在obj_mask中,那些有target_boxes的区域都设置为1.同理在noobj_mask中,有target_boxes的区域都设置为0
    # obj_mask第一维度最大本应为8(如果batch_size=8),但是这里不出意外的话应该会超过8,因为target_box会在同一张图片中有多个.
    # 这里obj_mask中的值如何才能算作1呢,就是target_boxes的坐标向下取整后和哪个grid坐标相同,target_boxes就属于那个grid里.
    # anchor这里也是一样target_boxes的长宽和哪个尺寸的anchor的iou最接近就属于哪个anchor(best_ind).
    # 以及这个target_boxes原本属于哪张图片的(i_in_batch),最后就由这四个值决定的.noobj_mask同理

    # Set noobj mask to zero where iou exceeds ignore threshold
    # ious.t():
    # shape: (n_boxes, num_anchor)
    # i: box id
    # b[i]: img index in that batch
    # E.g 假设有4个boxes，其所属图片在batch中的index为[0, 0, 1, 2], 即前2个boxes都属于本batch中的第0张图
    #     则b[0] = b[1] = 0 都应所属图片在batch中的index，即batch中的第几张图
    for i, anchor_ious in enumerate(ious.t()):
        # noobj是没有物体落在特征图中某个cell的索引，所以在初始化的时候初始化为1
        # 如果预测的IOU值过大，(大于阈值ignore_thres)时，那么可以认为这个cell是有物体的，要置0
        # 正例除外(与ground truth计算后IOU最大的检测框，但是IOU小于阈值，仍为正例)
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0   # 有目标

    # TODO 这部分写的注释有点多，来个总结：
    # 预测框分为三种：正例(positive), 负例(negative), 忽略样本(ignore)

    # 正例:任取一个ground truth，与4032个框全部计算IOU，IOU最大的预测框，即为正例。并且一个预测框，只能分配给一个ground truth。
    # 例如第一个ground truth已经匹配了一个正例检测框，那么下一个ground truth，就在余下的4031个检测框中，寻找IOU最大的检测框作为正例。
    # ground truth的先后顺序可忽略。正例产生置信度loss、检测框loss、类别loss。
    # 预测框为对应的ground truth box标签(需要反向编码，使用真实的x、y、w、h计算出[公式])；类别标签对应类别为1，其余为0；置信度标签为1。

    # 忽略样例：正例除外，与任意一个ground truth的IOU大于阈值（论文中使用0.5），则为忽略样例。忽略样例不产生任何loss。

    # 负例：正例除外(与ground truth计算后IOU最大的检测框，但是IOU小于阈值，仍为正例)，
    # 与全部ground truth的IOU都小于阈值（0.5），则为负例。负例只有置信度产生loss，置信度标签为0。

    # Coordinates
    # 因为YOLO的核心思想就是物体的的中心点落在哪一个方格（cell）中，那个方格就预测该物体。
    # 有人会问这里为什么没有使用sigmoid，如果你仔细看YoloLayer的前向传播(forward())，
    # 在使用build_targets函数处理前，就已经使用sigmoid函数处理过了
    # 真实值是gx，gy... 网络训练的是tx，ty...，所以这部分是这么变换的
    tx[b, best_n, gj, gi] = gx - gx.floor()        # x_offset (0, 1)
    ty[b, best_n, gj, gi] = gy - gy.floor()        # y_offset (0, 1)
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    # 是表明第b张图片，使用第best_n个anchors来预测 哪一类（target_labels）物体
    # 这里的target_labels可能是多个的。例如：狗和哈士奇，也就是多标签的意思
    # 这是一个标签掩膜,有target的那一类target_label为1    one-hot编码方式
    #         狗  飞机  船 哈士奇
    #  box1   1    0   0    1
    #  box2   0    1   0    0
    tcls[b, best_n, gj, gi, target_labels] = 1   # 标签掩膜
    # Compute label correctness and iou at best anchor
    # 这两个是为了评价用，不参与实际回归
    # pred_cls与target_label匹配上了，则class_mask所对应的grid_xy为1，否则为0
    # 所有预测中类别预测正确则class_mask对应位置为1
    # pred_cls[i_in_batch, best_ind, gj, gi] 是一个 (len(target),num_class)的数据.
    # 即网络在target_box位置预测的所有种类(16)的概率值  shape  -> len(target),16
    # pred_cls[i_in_batch, best_ind, gj, gi].argmax(-1) 代表网络在target_box位置预测的最大概率的类的索引(即max_class_index)
    # pred_cls --> (b, 3, 13, 13, 80)
    # class_mask 类掩膜 在target_boxes位置上预测的分类概率最大的那个class_id与target_boxes的class_id一致才会令该处的值为 1, 否则默认为 0
    # argmax(-1)得到预测分类的值，然后判断和真实值是否相等，得到了正确分类的index
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # pred_boxes[i_in_batch, best_ind, gj, gi]为(len(target),4)的tensor,这里只是计算网络在target_boxes位置预测的xywh与真实的xywh的iou
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    # tconf 这里进行float处理的原因是为了后面计算loss时和pred_box的float类型对齐
    tconf = obj_mask.float()   # target confidence
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    #  iou_scores --> 最好的那个anchor与target_box的IOU值
    #  class_mask --> 分类正确的索引值
    #  obj_mask --> 目标所在位置的最好anchor 值为1
    #  noobj_mask -->
    #  tx, ty, tw, th --> 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值
    #  tconf --> 目标置信度 --> 其实就是obj_mask换成了float