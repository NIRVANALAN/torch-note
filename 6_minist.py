import os

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn


torch.manual_seed(1)

# Hyper Params
EPOCH = 1
BATCH_SIZE = 64
LR = 1e-3
Download_Minist = False
# Minist digits dataset
if not (os.path.exists('./minist')) or not os.listdir('./minist/'):
	Download_Minist = True

train_data = torchvision.datasets.MNIST(
	root='./minist.',
	train=True,
	transform=torchvision.transforms.ToTensor(),  # PIL image to FloatTensor
	download=Download_Minist,
)

print(train_data.train_data.size())
print(train_data.train_labels.size())

plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
