import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init

class generator(Module):
    """The generator network inputs a vector of 1024 dimensions to generate an image with a resolution of 112 * 112.

    Args:
        dim(int): channel of training image
    """

    # initializers
    def __init__(self, dim=3):
        super(generator, self).__init__()
        self.fc = nn.Linear(1024, 7*7*256)
        self.fc_bn = nn.BatchNorm(256)
        self.deconv1 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)
        self.deconv1_bn = nn.BatchNorm(256)
        self.deconv2 = nn.ConvTranspose(256, 256, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm(256)
        self.deconv3 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)
        self.deconv3_bn = nn.BatchNorm(256)
        self.deconv4 = nn.ConvTranspose(256, 256, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm(256)
        self.deconv5 = nn.ConvTranspose(256, 128, 3, 2, 1, 1)
        self.deconv5_bn = nn.BatchNorm(128)
        self.deconv6 = nn.ConvTranspose(128, 64, 3, 2, 1, 1)
        self.deconv6_bn = nn.BatchNorm(64)
        self.deconv7 = nn.ConvTranspose(64 , dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # forward method
    def execute(self, input):
        x = self.fc(input).reshape((input.shape[0], 256, 7, 7))
        x = self.relu(self.fc_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.relu(self.deconv5_bn(self.deconv5(x)))
        x = self.relu(self.deconv6_bn(self.deconv6(x)))
        x = self.tanh(self.deconv7(x))

        return x


class discriminator(nn.Module):
    """The discriminator network inputs 112 * 112 images and outputs a number to indicate the credibility of the input image as a real image.

    Args:
        dim(int): channel of training image
    """

    # initializers
    def __init__(self, dim=3):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv(dim, 64, 5, 2, 2)
        self.conv2 = nn.Conv(64, 128, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm(128)
        self.conv3 = nn.Conv(128, 256, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm(256)
        self.conv4 = nn.Conv(256, 512, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm(512)
        self.fc = nn.Linear(512*7*7, 1)
        self.leaky_relu = nn.Leaky_relu()

    # forward method
    def execute(self, input):
        x = self.leaky_relu(self.conv1(input), 0.2)
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.reshape((x.shape[0], 512*7*7))
        x = self.fc(x)
        return x

def ls_loss(x, b):
    """Least square loss function.
    
    Args:
        x(jittor.array): Output vector of Generator
        b(bool): Indicates whether the image wants to be judged as true
    """
    mini_batch = x.shape[0]
    y_real_ = jt.ones((mini_batch,))
    y_fake_ = jt.zeros((mini_batch,))
    if b:
        return (x-y_real_).sqr().mean()
    else:
        return (x-y_fake_).sqr().mean()