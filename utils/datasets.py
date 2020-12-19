# -*- coding:utf-8 -*-
import glob
import random
import os, cv2
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def load_image(self, index):
    # loads 1 image from dataset
    img = self.imgs[index]
    if img is None:
        img_path = self.img_files[index].replace('\n', '')
        # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # img = cv2.imread(os.path.join(root_dir, img_path))  # BGR
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path
        r = self.img_size / max(img.shape)  # size ratio
        if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # _LINEAR fastest

    # Augment colorspace
    if self.augment:
        augment_hsv(img, hgain=0.10, sgain=0.5703, vgain=0.3174)
    return img


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = (np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1).astype(np.float32)  # random gains
    # np.random.uniform(-1, 1, 3)  在[-1, 1)内取三个数,float类型
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 关于random.uniform()的思考   uniform() 方法将随机生成下一个实数，它在[x, y]范围内
    # uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。
    # https://www.runoob.com/python/func-number-uniform.html
    img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128
    # base image with 4 tiles  128是灰色  新图先设置成4倍再resize回去，128是灰色
    indices = [index] + [random.randint(0, len(self.label_files) - 1) for _ in range(3)]
    # 3 additional image indices  随机取图像的索引，根据索引取图像。这一步再随机取三张图像，加上本身索引的这张图像共四张图像
    for i, index in enumerate(indices):
        # Load image
        img = load_image(self, index)

        h, w, _ = img.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Load labels
        label_path = self.label_files[index].replace('\n', '')
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                # 此时label --> [cls, x, y, w, h]  -->第一个类别的索引，中间点大小，宽高
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            # else:
                # labels = np.zeros((0,5), dtype=np.float32)

                labels4.append(labels)
    labels4 = np.concatenate(labels4, 0)   # 此时的labels4是个list，这个操作可以将list转成array

    # hyp = self.hyp
    # img4, labels4 = random_affine(img4, labels4,
    #                               degrees=hyp['degrees'],
    #                               translate=hyp['translate'],
    #                               scale=hyp['scale'],
    #                               shear=hyp['shear'])

    # Center crop
    a = s // 2
    img4 = img4[a:a + s, a:a + s]
    labels4[:, 1:] -= a      # 第一列为label的索引，这部分不改变，剩下的图像部分同时减去a
    return img4, labels4


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))    # raw
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        n = len(self.img_files)
        self.labels = [None] * n
        self.imgs = [None] * n

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # mosaic = True and self.augment  # 4 images
        # if mosaic:
        #     img, labels = load_mosaic(self, index)
        #     h, w, _ = img.shape
        # else:
        #     img = load_image(self, index)
        #     h, w, _ = img.shape

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # boxes[:, 0] => class id
            # Returns (xc, yc, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w        # newer center x
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h        # newer center y
            boxes[:, 3] *= w_factor / padded_w              # newer width
            boxes[:, 4] *= h_factor / padded_h              # newer height

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn_raw(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        max_targets = max([targets[i].size(0) for i in range(len(targets))])
        padded_targets = list()

        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[:, 0] = i
                absent = max_targets - boxes.size(0)
                if absent > 0:
                    boxes = torch.cat((boxes, torch.zeros((absent, 6))), 0)
                padded_targets.append(boxes)
        targets = [boxes for boxes in padded_targets]
        targets = torch.cat(targets, 0)
        # select new image size every 10 batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # resize image(pad-to-square) to new size
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
