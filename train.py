import jittor as jt
from jittor.dataset.mnist import MNIST
from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform
from model_lsgan import *
import os
import numpy as np
import matplotlib.pyplot as plt

jt.flags.use_cuda = 1

# 使用 MNIST 或者 CelebA数据集进行训练
# task="CelebA"
task="MNIST"

# 结果图片存储路径
save_path = "./results_img"
if not os.path.exists(save_path):
    os.mkdir(save_path)
fixed_z_ = jt.init.gauss((5 * 5, 1024), 'float')    # fixed noise
def save_result(num_epoch, G , path = 'result.png'):
    """Use the current generator to generate 5*5 pictures and store them.

    Args:
        num_epoch(int): current epoch
        G(generator): current generator
        path(string): storage path of result image
    """

    z_ = fixed_z_
    G.eval()
    test_images = G(z_)
    G.train()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i in range(size_figure_grid):
        for j in range(size_figure_grid):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if task=="MNIST":
            ax[i, j].imshow((test_images[k, 0].data+1)/2, cmap='gray')
        else:
            ax[i, j].imshow((test_images[k].data.transpose(1, 2, 0)+1)/2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


# 批大小
batch_size = 128
# 学习率
lr = 0.0002
# 训练轮数
train_epoch = 50 if task=="MNIST" else 50
# 训练图像标准大小
img_size = 112
# Adam优化器参数
betas = (0.5,0.999)
# 数据集图像通道数，MNIST为1，CelebA为3
dim = 1 if task=="MNIST" else 3

if task=="MNIST":
    transform = transform.Compose([
        transform.Resize(size=img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])
    train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
    eval_loader = MNIST(train=False, transform = transform).set_attrs(batch_size=batch_size, shuffle=True)
elif task=="CelebA":
    transform = transform.Compose([
        transform.Resize(size=img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dir = './data/celebA_train'
    train_loader = ImageFolder(train_dir).set_attrs(transform=transform, batch_size=batch_size, shuffle=True)
    eval_dir = './data/celebA_eval'
    eval_loader = ImageFolder(eval_dir).set_attrs(transform=transform, batch_size=batch_size, shuffle=True)

G = generator (dim)
D = discriminator (dim)
G_optim = jt.nn.Adam(G.parameters(), lr, betas=betas)
D_optim = jt.nn.Adam(D.parameters(), lr, betas=betas)

def train(epoch):
    for batch_idx, (x_, target) in enumerate(train_loader):
        mini_batch = x_.shape[0]

        # train discriminator
        D_result = D(x_)
        D_real_loss = ls_loss(D_result, True)
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result_ = D(G_result)
        D_fake_loss = ls_loss(D_result_, False)
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.sync()
        D_optim.step(D_train_loss)

        # train generator
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = ls_loss(D_result, True)
        G_train_loss.sync()
        G_optim.step(G_train_loss)
        if (batch_idx%100==0):
            print("train batch_idx",batch_idx,"epoch",epoch)
            print('  D training loss =', D_train_loss.data.mean())
            print('  G training loss =', G_train_loss.data.mean())

def validate(epoch):
    D_losses = []
    G_losses = []
    G.eval()
    D.eval()
    for batch_idx, (x_, target) in enumerate(eval_loader):
        mini_batch = x_.shape[0]
        
        # calculation discriminator loss
        D_result = D(x_)
        D_real_loss = ls_loss(D_result, True)
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result_ = D(G_result)
        D_fake_loss = ls_loss(D_result_, False)
        D_train_loss = D_real_loss + D_fake_loss
        D_losses.append(D_train_loss.data.mean())

        # calculation generator loss
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = ls_loss(D_result, True)
        G_losses.append(G_train_loss.data.mean())
    G.train()
    D.train()
    print("validate: epoch",epoch)
    print('  D validate loss =', np.array(D_losses).mean())
    print('  G validate loss =', np.array(G_losses).mean())

for epoch in range(train_epoch):
    print ('number of epochs', epoch)
    train(epoch)
    validate(epoch)
    result_img_path = './results_img/' + task + str(epoch) + '.png'
    save_result(epoch, G, path=result_img_path)
