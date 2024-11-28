import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join
import cv2

warnings.filterwarnings('ignore')



class goproDataset(Dataset):
    def __init__(self, spikedatasetPath, imgPath, gtimgPath, sampleFile, datarand = False, cropsize = 0):
        self.spike_path = spikedatasetPath
        self.img_path = imgPath
        self.gt_path = gtimgPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.cropsize = cropsize
        self.datarand = datarand

    def __getitem__(self, index):
        # inputIndex = self.samples[index]
        if self.datarand:
            inputIndex = self.samples[np.random.randint(self.samples.shape[0])]
        else:
            inputIndex = self.samples[index]
        # print(self.spike_path + str(inputIndex.item()) + '.pt')

        inputSpikes = torch.load(self.spike_path + str(inputIndex.item()) + '.pt')
        inputSpikes = inputSpikes.permute(0, 1, 3, 2)
        # inputSpikes = snn.io.readplus2Dspikes(
        #     self.spike_path + str(inputIndex.item()) + '.pbs2'
        # ).toSpikeTensor(torch.zeros((2, 720, 1280, self.nTimeBins)),
        #                 samplingTime=self.samplingTime)
        inputImg = readImgs(self.img_path + str(inputIndex.item()) + '.png', channel=3)
        # inputImg = cv2.resize(inputImg, (346, 260), interpolation=cv2.INTER_LINEAR)
        input_Img = transforms.ToTensor()(inputImg)
        # data_mean = input_Img.mean()
        # data_std = input_Img.std()
        # input_Img = transforms.Normalize(data_mean, data_std)(input_Img)
        gtImg    = readImgs(self.gt_path + str(inputIndex.item()) + '.png', channel=3)
        gt_Img = transforms.ToTensor()(gtImg)
        # data_mean = gt_Img.mean()
        # data_std = gt_Img.std()
        # gt_Img = transforms.Normalize(data_mean, data_std)(gt_Img)

        ps = self.cropsize
        _, w, h = gt_Img.shape
        if self.cropsize and w > ps and h > ps:
            hh, ww = gt_Img.shape[1], gt_Img.shape[2]
            rr     = random.randint(0, hh-ps)
            cc     = random.randint(0, ww-ps)

            # Crop patch
            input_Img = input_Img[:, rr:rr+ps, cc:cc+ps]
            gt_Img = gt_Img[:, rr:rr+ps, cc:cc+ps]
            inputSpikes = inputSpikes[:, :, rr:rr+ps, cc:cc+ps]

        # apply to resnet backbone
        # if Img.shape[0] == 1:
        #     Img = torch.cat([Img, Img, Img], dim=0)

        return inputSpikes, input_Img, gt_Img, inputIndex

    def __len__(self):
        return self.samples.shape[0]


class REBlurDataset(Dataset):
    def __init__(self, spikedatasetPath, imgPath, gtimgPath, sampleFile, datarand = False, cropsize = 0):
        self.spike_path = spikedatasetPath
        self.img_path = imgPath
        self.gt_path = gtimgPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.cropsize = cropsize
        self.datarand = datarand

    def __getitem__(self, index):
        # inputIndex = self.samples[index]
        if self.datarand:
            inputIndex = self.samples[np.random.randint(self.samples.shape[0])]
        else:
            inputIndex = self.samples[index]

        inputSpikes = torch.load(self.spike_path + str(inputIndex.item()) + '.pt')
        inputSpikes = inputSpikes[:,:,:256,:]
        # inputSpikes = inputSpikes.permute(0, 1, 3, 2)

        inputImg = readImgs(self.img_path + str(inputIndex.item()) + '.png', channel=3)[:256,:,:]
        input_Img = transforms.ToTensor()(inputImg)

        gtImg    = readImgs(self.gt_path + str(inputIndex.item()) + '.png', channel=3)[:256,:,:]
        gt_Img = transforms.ToTensor()(gtImg)

        ps = self.cropsize
        _, w, h = gt_Img.shape
        if self.cropsize and w > ps and h > ps:
            hh, ww = gt_Img.shape[1], gt_Img.shape[2]
            rr     = random.randint(0, hh-ps)
            cc     = random.randint(0, ww-ps)

            # Crop patch
            input_Img = input_Img[:, rr:rr+ps, cc:cc+ps]
            gt_Img = gt_Img[:, rr:rr+ps, cc:cc+ps]
            inputSpikes = inputSpikes[:, :, rr:rr+ps, cc:cc+ps]

        return inputSpikes, input_Img, gt_Img, inputIndex

    def __len__(self):
        return self.samples.shape[0]

def build_gopro(spikedatasetPath, bulrPath, gtimgPath, train_index_file, test_index_file, crop = 0):
    train_dataset = goproDataset(spikedatasetPath=spikedatasetPath,
                                 imgPath=bulrPath,
                                 gtimgPath=gtimgPath,
                                 sampleFile=train_index_file,
                                 datarand=False,
                                 cropsize=crop)
    test_dataset = goproDataset(spikedatasetPath=spikedatasetPath,
                                imgPath=bulrPath,
                                gtimgPath=gtimgPath,
                                sampleFile=test_index_file,
                                datarand=False,
                                cropsize=0)
    return train_dataset, test_dataset


def build_REBlur(spikedatasetPath, bulrPath, gtimgPath, train_index_file, test_index_file, crop = 0):
    train_dataset = REBlurDataset(spikedatasetPath=spikedatasetPath,
                                 imgPath=bulrPath,
                                 gtimgPath=gtimgPath,
                                 sampleFile=train_index_file,
                                 datarand=False,
                                 cropsize=0)
    test_dataset = REBlurDataset(spikedatasetPath=spikedatasetPath,
                                imgPath=bulrPath,
                                gtimgPath=gtimgPath,
                                sampleFile=test_index_file,
                                datarand=False,
                                cropsize=0)
    return train_dataset, test_dataset



def readImgs(filename, channel = 1):
    '''
    Reads two dimensional image file and returns an img tensor.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> Img = spikeFileIO.readImgs(file_path, channel=1)
    '''

    # Img = plt.imread(filename)
    if channel == 1:
        Img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    elif channel == 3:
        Img = cv2.imread(filename)
    else:
        Img = None

    return Img

if __name__ == '__main__':
    train_set, test_set = build_gopro()
