from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
from getDataLoader import BaseDataProvider
import scipy.io as sio

class DataProvider(BaseDataProvider):
    def __init__(self, inputSize, fineSize, path, labelPath, mode=None):
        super(DataProvider, self).__init__()
        self.inputSize  = inputSize
        self.fineSize   = fineSize
        self.path       = path
        self.labelPath  = labelPath
        self.mode       = mode
        self.data_idx   = -1
        self.n_data     = self._load_data()

    def _load_data(self):
        self.imageNum = []

        datapath = os.path.join(self.path)
        dataFiles  = sorted(os.listdir(datapath))
        for isub, dataName in enumerate(dataFiles):
            self.imageNum.append(os.path.join(datapath, dataName))
        label = np.load(os.path.join(self.labelPath, 'atlas_norm.npz'))
        self.label = label['vol'] #['seg']

        if self.mode == "train":
            np.random.shuffle(self.imageNum)
        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        dataPath = self.imageNum[self.data_idx]
        data_ = sio.loadmat(dataPath)
        data = data_['data_affine']
        return data, self.label, dataPath

    def _augment_data(self, data, label):
        if self.mode == "train":
            # Rotation 90
            op = np.random.randint(0, 4)
            data, label = np.rot90(data, op), np.rot90(label, op)

            # Flip horizon / vertical
            op = np.random.randint(0, 3)
            if op < 2:
                data, label = np.flip(data, op), np.flip(label, op)

        return data, label


