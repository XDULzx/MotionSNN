import struct
import numpy as np
import scipy.misc
import h5py
import glob
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat
import sys
import cv2


train_filename = "DvsGesture/dvs_gestures_dense_train.pt"
test_filename = "DvsGesture/dvs_gestures_dense_test.pt"

mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird' ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'    ,
            6 :'frog'   ,
            7 :'horse'       ,
            8 :'ship'      ,
            9 :'truck'     }


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '{}.pt'.format(index))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


def gather_addr(directory, start_id, end_id):
    fns = []
    for i in range(start_id, end_id):
        for class0 in mapping.keys():
            file_name = directory + '/' + mapping[class0] + '/' + "{}".format(i) + '.mat'
            fns.append(file_name)
    return fns


def events_to_frames(filename, t):
    frames = np.zeros((t, 2, 1280, 720))
    events = readplus2Dspikes(filename) # N,4 [t, x, y, p] p=0,1
    events[:,0] = events[:,0] - events[0,0] + 1
    d_t = int(events[:,0].max().item() / t)

    for i in range(t):
        fx = events[:, 1][np.argwhere((events[:, 0] > i * d_t) & (events[:, 0] < (i +1)*d_t ))]
        fy = events[:, 2][np.argwhere((events[:, 0] > i * d_t) & (events[:, 0] < (i +1)*d_t ))]
        fp = events[:, 3][np.argwhere((events[:, 0] > i * d_t) & (events[:, 0] < (i +1)*d_t ))]

        frames[i, fp, fx, fy] += 1  # events[r1:r2, 0]

    for i in range(t):
        frames[i, :, :, :] = frames[i, :, :, :] / np.max(frames[i, :, :, :])

    return frames

def readplus2Dspikes(filename):
    with open(filename, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = np.asarray([x for x in inputByteArray])
    xEvent = (inputAsInt[0::7] << 8) | (inputAsInt[1::7])
    yEvent = (inputAsInt[2::7] << 8) | (inputAsInt[3::7])
    pEvent = inputAsInt[4::7] >> 7
    tEvent = ((inputAsInt[4::7] << 16) | (inputAsInt[5::7] << 8) | (inputAsInt[6::7])) & 0x7FFFFF

    return torch.tensor(np.concatenate((tEvent[:,np.newaxis], xEvent[:,np.newaxis], yEvent[:,np.newaxis], pEvent[:,np.newaxis]), axis=1))

def create_npy(inpath='gopro_bs2', outpath = 'gopro_out'):
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    dir = np.sort(os.listdir(inpath))

    print("processing data...")
    key = 1
    T = 10
    for file in dir:
        filename = os.path.join(inpath, file)
        findex = file.replace('pbs2','')
        outfilename = os.path.join(outpath, findex + 'pt')
        frames = events_to_frames(filename, t=T)
        if key % 1 == 0:
            print(file +  ' '+ findex + 'pt' +" Data {:.4f}% complete\t\t".format(100 * key / dir.size))
        key += 1
        torch.save(torch.Tensor(frames), outfilename)


if __name__ == "__main__":
    eventbs2_path   = "/gopropbs2/data/events"
    out_path        = "/gopropbs2_new/data/ptevents_T10"
    create_npy(inpath=eventbs2_path, outpath=out_path)
