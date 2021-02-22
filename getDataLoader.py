from __future__ import print_function, division, absolute_import, unicode_literals
import torch

class BaseDataProvider(object):

    def _load_data_and_label(self):
        data, label, path = self._next_data()
        data, label = self._augment_data(data, label)

        data = data.transpose(2, 0, 1).astype(float)
        labels = label.transpose(2, 0, 1).astype(float)
        nd = data.shape[0]
        nw = data.shape[1]
        nh = data.shape[2]
        return path, data.reshape(1, 1, nd, nw, nh), labels.reshape(1, 1, nd, nw, nh)

    def _toTorchFloatTensor(self, img):
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        path, data, labels = self._load_data_and_label()
        P = []

        X = self._toTorchFloatTensor(data)
        Y = self._toTorchFloatTensor(labels)
        P.append(path)

        return X, Y, P