# -*- coding:utf-8 -*-
# author:栗山未来ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:aaa.py
# software: PyCharm

import numpy as np


class DataGenerator(object):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __call__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for img_idx in indices:
            img, img_meta, bbox, label = self.dataset[img_idx]
            yield img, img_meta, bbox, label
