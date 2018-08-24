""" Fully Convolution Network: pretrained- VGG16 """
import torch
import torch.nn as nn
from torchvision.models import vgg16

class Vgg16Part(nn.Module):
    def __init__(self):
        super(Vgg16Part, self).__init__()
        self.pre_vgg16 = vgg16(pretrained=False) #FIXME
        self.l3 = []
        self.l4 = []
        self.l5 = []
        self.transfer()
    
    def transfer(self):
        layers = self.pre_vgg16._modules['features']
        for i in range(0, 17):
            self.l3.append(layers[i])
        self.l3 = nn.Sequential(*self.l3)
        for i in range(17, 24):
            self.l4.append(layers[i])
        self.l4 = nn.Sequential(*self.l4)
        for i in range(24, 31):
            self.l5.append(layers[i])
        self.l5 = nn.Sequential(*self.l5)
    
    def forward(self, x):
        pool3 = self.l3(x)
        pool4 = self.l4(pool3)
        pool5 = self.l5(pool4)
        return pool3, pool4, pool5
        

class FCN8(nn.Module):
    def __init__(self, n_classes=21):
        super(FCN8, self).__init__()
        self.pretrained = Vgg16Part()
        self.n_classes = n_classes

        # Convolutionized FC Block
        # Preserve feature map size
        self.fc_to_conv1 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU())
        self.fc_to_conv2 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU())

        # Prediction Block
        self.score_last = nn.Conv2d(4096, n_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=1)

        # Upsample Block
        self.up2x = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.up8x = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        pool3, pool4, pool5 = self.pretrained(x) # 1/8, 1/16 1/32

        # pool5 path
        pool5 = self.fc_to_conv1(pool5)
        pool5 = self.fc_to_conv2(pool5)
        pool5 = self.score_last(pool5)
        up2_pred = self.up2x(pool5)

        # pool4 path
        pool4_pred = self.score_pool4(pool4)

        assert up2_pred.size() == pool4_pred.size(), "Size mismatched-{}/{}" \
                                                      .format(up2_pred.size(), pool4_pred.size())
        up2_pred += pool4_pred
        up2_pred = self.up2x(up2_pred)

        # pool3 path
        pool3_pred = self.score_pool3(pool3)

        assert up2_pred.size() == pool3_pred.size(), "Size mismatched-{}/{}" \
                                                      .format(up2_pred.size(), pool3_pred.size())
        up2_pred += pool3_pred

        # Output
        out = self.up8x(up2_pred)
        return out


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 256))
    model = FCN8()
    out = model(x)
    print(out.size())