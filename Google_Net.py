# imports
import torch
import torch.nn as nn


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = Conv_Block(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3))

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_Block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # This order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_Block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_Block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = nn.Sequential(
            self.inception3a,
            self.inception3b,
            self.maxpool3
        )

        self.inception4a = Inception_Block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_Block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_Block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_Block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_Block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4 = nn.Sequential(
            self.inception4a,
            self.inception4b,
            self.inception4c,
            self.inception4d,
            self.inception4e,
            self.maxpool4
        )

        self.inception5a = Inception_Block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_Block(832, 384, 192, 384, 48, 128, 128)
        self.inception5 = nn.Sequential(
            self.inception5a,
            self.inception5b
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3(x)

        x = self.inception4(x)

        x = self.inception5(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_Block, self).__init__()

        self.branch1 = Conv_Block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv_Block(in_channels, red_3x3, kernel_size=1),
            Conv_Block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            Conv_Block(in_channels, red_5x5, kernel_size=1),
            Conv_Block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv_Block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], 1)


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), **kwargs):
        super(Conv_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = GoogleNet()
    print(model(x).shape)
