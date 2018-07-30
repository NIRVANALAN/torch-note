import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

if not os.path.exists('./dc_img'):
	os.mkdir('./dc_img')
	'''
	Normalize
	``input[channel] = (input[channel] - mean[channel]) / std[channel]``
	'''


def to_img(x):
	out = 0.5 * (x + 1)
	out = out.clamp(0, 1)
	out = out.view(1, -1, 28, 28)
	return out


batch_size = 128
num_epoch = 100
z_dimension = 100  # noise dimension

img_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

mnist = datasets.MNIST(root='./mnist', train=True, transform=img_transform, download=False)
dataloader = torch.utils.data.Dataloader(datasets=mnist, batch_size=batch_size, shuffle=True)

class discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		