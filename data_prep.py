import os
import sys
import numpy as np
import copy
import math
import scipy.io as sio
import scipy.misc as smc
import glob
import scipy.ndimage


class FlowDataset():
    def __init__(self, cachepath, npoint=512, nmask=10):
        self.cachepath = cachepath
        self.npoint = npoint
        self.nmask = nmask
        self.data = sio.loadmat(cachepath)
        self.np_rand = np.random.RandomState(0)

    def generate_3d(self):
        """Generate a 3D random rotation matrix.
        Returns:
            np.matrix: A 3D rotation matrix.
        """
        x1, x2, x3 = self.np_rand.rand(3)
        R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                       [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                       [0, 0, 1]])
        v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sqrt(1 - x3)]])
        H = np.eye(3) - 2 * v * v.T
        M = -H * R
        return M

    def __getitem__(self, index):

        pc1 = copy.deepcopy(self.data['pc1'][index])
        pc2 = copy.deepcopy(self.data['pc2'][index])

        pc1, pc2 = map(np.asarray, [pc1, pc2])

        return pc1, pc2

    def __len__(self):
        return self.data['pc1'].shape[0]


class SegDataset():
    def __init__(self, cachepath, npoint=512, nmask=10, relrot=True):
        self.cachepath = cachepath
        self.npoint = npoint
        self.nmask = nmask
        self.relrot = relrot
        self.data = sio.loadmat(cachepath)

    def generate_3d(self):
        """Generate a 3D random rotation matrix.
        Returns:
            np.matrix: A 3D rotation matrix.
        """
        x1, x2, x3 = np.random.rand(3)
        R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                       [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                       [0, 0, 1]])
        v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sqrt(1 - x3)]])
        H = np.eye(3) - 2 * v * v.T
        M = -H * R
        return M

    def __getitem__(self, index):
        pc1 = copy.deepcopy(self.data['pc1'][index])
        pc2 = copy.deepcopy(self.data['pc2'][index])
        flow12 = copy.deepcopy(self.data['flow12'][index])
        momasks = copy.deepcopy(self.data['momasks'][index])

        permidx = np.random.permutation(pc1.shape[0])[:self.npoint]
        pc1 = pc1[permidx, :]
        flow12 = flow12[permidx, :]
        momasks = np.eye(self.nmask)[np.minimum(
            momasks, self.nmask-1)[permidx].astype('int32')]
        permidx2 = np.random.permutation(pc2.shape[0])[:self.npoint]
        pc2 = pc2[permidx2, :]

        # global transform
        R0 = self.generate_3d()
        pc1 = np.matmul(pc1, R0)
        pc2 = np.matmul(pc2, R0)
        flow12 = np.matmul(flow12, R0)

        if self.relrot:
            # relative transform
            R1 = 0.5*self.generate_3d()+np.eye(3)
            u1, _, vh1 = np.linalg.svd(R1, full_matrices=True)
            R1 = np.matmul(u1, vh1)
            flow12 = pc1+flow12-np.matmul(pc1, R1)
            pc1 = np.matmul(pc1, R1)

        vismask = np.zeros(self.npoint)
        return pc1, pc2, flow12, vismask, momasks

    def __len__(self):
        return self.data['pc1'].shape[0]


class SynTestDataset():
    def __init__(self, cachepath, npoint=512):
        self.cachepath = cachepath
        self.npoint = npoint
        self.data = sio.loadmat(cachepath)

    def __getitem__(self, index):
        pc1 = copy.deepcopy(self.data['pc1'][index])
        pc2 = copy.deepcopy(self.data['pc2'][index])
        flow12 = copy.deepcopy(self.data['flow12'][index])
        seg1 = copy.deepcopy(self.data['seg1'][index]).astype('int32')-1
        if self.npoint < pc1.shape[0]:
            permidx = np.random.permutation(pc1.shape[0])[:self.npoint]
            pc1 = pc1[permidx, :]
            flow12 = flow12[permidx, :]
            seg1 = seg1[permidx]
            permidx2 = np.random.permutation(pc2.shape[0])[:self.npoint]
            pc2 = pc2[permidx2, :]
        return pc1, pc2, flow12, seg1

    def __len__(self):
        return self.data['pc1'].shape[0]


if __name__ == '__main__':
    D = FlowDataset(cachepath='data/flow_validation.mat', npoint=512)
    print(len(D))
    pc1, pc2, flow12, vismask, momasks = D[0]
    print(pc1.shape, pc2.shape, flow12.shape, vismask.shape, momasks.shape)

    D = SegDataset(cachepath='data/seg_validation.mat')
    print(len(D))
    pc1, pc2, flow12, vismask, momasks = D[0]
    print(pc1.shape, pc2.shape, flow12.shape, vismask.shape, momasks.shape)
