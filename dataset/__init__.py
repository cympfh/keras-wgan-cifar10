import math
import random

import numpy
from keras.datasets import cifar10
from keras import utils
from PIL import Image


class Sequence(utils.Sequence):

    def __init__(self, X, batch_size, indices=None, test=False):
        self.X = X
        self.batch_size = batch_size
        self.indices = indices or list(range(len(X)))
        self.test = test

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        test = self.test
        begin = idx * self.batch_size
        end = begin + self.batch_size
        batch_idx = self.indices[begin: end]
        batch_x = self.X[batch_idx]

        # augmentation
        if not test:
            batch_x += numpy.random.normal(size=batch_x.shape, scale=0.01)

        return batch_x


def batch_generator(batch_size, validation_split=0.1, test=False):
    (x_train, _), (x_test, _) = cifar10.load_data()
    if not test:
        X = x_train.astype('f') / 128.0 - 1.0
        num = len(X)
        indices = list(range(num))
        random.shuffle(indices)
        num_valid = int(num * validation_split)
        indices_train = indices[num_valid:]
        # indices_valid = indices[:num_valid]
        seq_train = Sequence(X, batch_size, indices=indices_train)
        # seq_valid = Sequence(X, batch_size, indices=indices_valid)
        return seq_train
    else:
        X = x_test.astype('f') / 128.0 - 1.0
        return Sequence(X, batch_size, test=True)


def array2images(x):
    imgs = []
    x = ((x + 1.0) * 128.0).astype(numpy.uint8)
    for i in range(len(x)):
        img = Image.fromarray(x[i, :, :, :])
        imgs.append(img)
    return imgs
