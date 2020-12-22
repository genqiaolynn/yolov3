import torch, cv2
import torch.nn.functional as F
import numpy as np
from torchvision import utils as vutils


def horisontal_flip(images, targets):
    '''
    :param images: pad之后的图像  tensor
    :param targets: 标注好的box  [x, y, w, h]
    :return: 镜像后的图像， 标签和box
    '''
    # 图像进行水平翻转
    # torch.flip(input,dim):第一个参数是输入，第二个参数是输入的第几维度，按照维度对输入进行翻转
    # image --> [B, C, H, W]   torch.flip(image, [-1]) 对宽进行翻转
    images = torch.flip(images, [-1])

    # targets是N行6列的数组，第0列是0， 第1到5列是GT的类别加上归一化后的box大小
    # targets[:, 2]是xmin
    targets[:, 2] = 1 - targets[:, 2]
    # 新的targets是水平翻转之后的。疑问：xmin变化了，那xmax为什么不翻转？？  targets[:, 4] = 1 - targets[:, 4]
    # 可能的原因:以为这里的x是box的中心坐标，减完还是中心坐标和宽高之间。以上的写法是xmin和xmax之间的转化。两者之间是有区别的。
    return images, targets
