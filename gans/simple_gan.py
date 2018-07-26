import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

if not os.path.exists('./img'):
    os.mkdir('./img')


# ?
def to_img(x):
    out = 0.5 * x + 1
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

# image pre processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# MNIST dataset load
mnist = datasets.MNIST(root='./mnist',
                       train=True, transform=img_transform, download=False)
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


D = discriminator()
G = generator()

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=3e-4)
g_optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)

# start training
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # ==============train discriminator============
        # define real/fake image, label
        img = img.view(num_img, -1)
        real_img = Variable(img).cuda()
        # real img's label is 1
        real_label = Variable(torch.ones(num_img)).cuda()
        # fake img's label is 0
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_score = real_out

        # compute loss for fake img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_score = fake_out

        # bp amd optimize
        d_loss = d_loss_real + d_loss_fake
        # zero the grad manually
        d_optimizer.zero_grad()
        # important
        d_loss.backward()
        d_optimizer.step()
        # ======== train generator=========
        # loss of fake img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)  # make generator generate well

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss:{:.6f}, g_loss:{:.6f}, D real:{:.6f}, D fake :{:.6f}'.format(epoch, num_epoch,
                                                                                                     d_loss.data[0],
                                                                                                     g_loss.data[0],
                                                                                                     real_score.data.mean(),
                                                                                                     fake_score.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './img/real_images.png')
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')
