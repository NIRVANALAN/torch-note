import pickle
import numpy as np
import os
import torchvision as tv
import torchvision.transforms as transforms
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as functional
from multiprocessing import freeze_support


class Net(nn.Module):
    def __init__(self):
        super()


if __name__ == '__main__':
    freeze_support()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # step 1
    train_set = ds.CIFAR10(root='./data', train=True, transform=transform, target_transform=None, download=False)
    print('train_set len:', len(train_set))
    # step 2
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

