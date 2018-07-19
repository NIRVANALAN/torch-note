from multiprocessing import freeze_support

from torch.autograd import Variable  #
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision import datasets as ds
from torch.utils.data import DataLoader

if __name__ == '__main__':
	freeze_support()
	device = 'cuda'
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	# step 1
	train_set = ds.CIFAR10(root='./data', train=True, transform=transform, target_transform=None, download=False)
	print('train_set len:', len(train_set))
	# step 2
	train_loader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=2)
	test_set = ds.CIFAR10(root='./data',train=False, transform=transform)
	test_loader = DataLoader(test_set, batch_size=4, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()  # nn.Module.__init__(self)
			self.conv1 = nn.Conv2d(3, 6, 5)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(16 * 5 * 5, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)
		
		def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = x.view(-1, 16 * 5 * 5)
			
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x
	
	
	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	
	for epoch in range(3):
		running_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			inputs, labels = inputs.cuda(), labels.cuda()
			inputs, labels = Variable(inputs), Variable(labels)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			optimizer.step()  # optimize the params
			
			running_loss += loss.data[0]
			if i % 1000 == 1:
				print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 1000))
				running_loss = 0.0
	print('Finished Training')

	dataiter = iter(test_loader)
	images, labels = dataiter.next()
	