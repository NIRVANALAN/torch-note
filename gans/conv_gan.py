import os
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

if __name__ == '__main__':
	freeze_support()
	if not os.path.exists('./dc_img'):
		os.mkdir('./dc_img')
		'''
		Normalize
		``input[channel] = (input[channel] - mean[channel]) / std[channel]``
		'''
	
	
	def to_img(x):
		out = 0.5 * (x + 1)
		out = out.clamp(0, 1)
		out = out.view(-1, 1, 28, 28)
		return out
	
	
	'''
	 x.clamp(0,1)
	Out[4]:
	tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
	        [ 0.0000,  0.0000,  0.0000,  0.0000],
	        [ 0.0548,  1.0000,  1.0000,  0.4647],
	        [ 0.0000,  0.0000,  0.0000,  0.0000]])
	
	In [5]: x
	Out[5]:
	tensor([[ 2.2048, -0.7127, -1.2530, -1.0886],
	        [-0.7682, -1.5979, -0.3934, -0.3327],
	        [ 0.0548,  1.0417,  1.4011,  0.4647],
	        [-0.0794, -0.9124, -1.1829, -0.0139]])
	'''
	
	batch_size = 128
	num_epoch = 100
	z_dimension = 100  # noise dimension
	
	img_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	
	mnist = datasets.MNIST(root='./mnist', train=True, transform=img_transform, download=False)
	dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, num_workers=4)
	
	
	class discriminator(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = nn.Sequential(
				nn.Conv2d(1, 32, 5, padding=2),  # batch 32 28 28
				nn.LeakyReLU(0.2, True),
				nn.AvgPool2d(2, stride=2),  # batch 32 14 14
			)
			self.conv2 = nn.Sequential(
				nn.Conv2d(32, 64, 5, padding=2),
				nn.LeakyReLU(0.2, True),
				nn.AvgPool2d(2, stride=2)  # batch 64 7 7
			)
			self.fc = nn.Sequential(
				nn.Linear(64 * 7 * 7, 1024),
				nn.LeakyReLU(0.2, True),
				nn.Linear(1024, 1),
				nn.Sigmoid()
			)
		
		def forward(self, x):
			x = self.conv1(x)
			x = self.conv2(x)
			x = x.view(x.size(0), -1)  # flatten the tensor
			x = self.fc(x)
			return x
	
	
	class generator(nn.Module):
		def __init__(self, input_size, num_feature):
			super().__init__()
			self.fc = nn.Linear(input_size, num_feature)  # batch 1*56*56
			self.br = nn.Sequential(
				nn.BatchNorm2d(1),  #####
				nn.ReLU(True)
			)
			self.downsample1 = nn.Sequential(
				nn.Conv2d(1, 50, 3, padding=1),  # batch 50 56 56
				nn.BatchNorm2d(50),
				nn.ReLU(True)
			)
			self.downsample2 = nn.Sequential(
				nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
				nn.BatchNorm2d(25),
				nn.ReLU(True)
			)
			self.downsample3 = nn.Sequential(
				nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
				nn.Tanh()
			)
		
		def forward(self, x):
			x = self.fc(x)
			x = x.view(x.size(0), 1, 56, 56)
			x = self.br(x)
			x = self.downsample1(x)
			x = self.downsample2(x)
			x = self.downsample3(x)
			return x
	
	
	if torch.cuda.is_available():
		D = discriminator().cuda()
		G = generator(z_dimension, 3136).cuda()
	
	criterion = nn.BCELoss()
	d_optimizer = torch.optim.Adam(D.parameters(), lr=3e-4)
	g_optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)
	
	# train
	
	for epoch in range(num_epoch):
		for i, (img, _) in enumerate(dataloader):  # GANs 中无需label 用_接收dataloader中传回的值，但后面不使用
			num_img = img.size(0)
			# ====== train discriminator =========
			real_img = Variable(img).cuda()
			real_label = Variable(torch.ones(num_img)).cuda()
			fake_label = Variable(torch.zeros(num_img)).cuda()
			
			# loss for real
			real_out = D(real_img)
			d_loss_real = criterion(real_out, real_label)
			real_scores = real_out
			
			# loss for fake
			z = Variable(torch.randn(num_img, z_dimension)).cuda()
			fake_img = G(z)
			fake_out = D(fake_img)
			d_loss_fake = criterion(fake_out, fake_label)
			fake_scores = fake_out
			# bp amd optimization
			d_loss = d_loss_real + d_loss_fake
			d_optimizer.zero_grad()
			d_loss.backward()
			d_optimizer.step()
			
			# ======train generator ========
			z = Variable(torch.randn(num_img, z_dimension)).cuda()
			fake_img = G(z)
			output = D(fake_img)
			g_loss = criterion(output, real_label)
			# bp optim
			g_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()
			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
				      'D real: {:.6f}, D fake: {:.6f}'
				      .format(epoch, num_epoch, d_loss.data[0], g_loss.data[0],
				              real_scores.data.mean(), fake_scores.data.mean()))
		if epoch == 0:
			real_images = to_img(real_img.cpu().data)
			save_image(real_images, './dc_img/real_images.png')
		fake_images = to_img(fake_img.cpu().data)
		save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch + 1))
	torch.save(G.state_dict(), './conv_gans_generator.pth')
	torch.save(D.state_dict(), './conv_gans_discriminator.pth')
