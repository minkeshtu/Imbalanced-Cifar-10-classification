'''
Form the paper "Inverted Residuals and Linear Bottlenecks: 
Mobile Networks for Classification, Detection and Segmentation".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class Mobilenetv2_linear_bottleneck_block(nn.Module):
    '''Linear Botleneck block -> expand + depthwise + pointwise
    For stride = 1 -> shortcut connection as well
    '''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Mobilenetv2_linear_bottleneck_block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class Mobilenetv2_base(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1), # NOTE: changed stride 2 -> 1 from original paper
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(Mobilenetv2_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False) # NOTE: changed conv1 stride 2 -> 1 from original paper
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layers = self._make_conv_layers(in_planes=32)

    def _make_conv_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Mobilenetv2_linear_bottleneck_block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        features = self.conv_layers(out)
        return features


if __name__ == '__main__':
    model = Mobilenetv2_base()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torchsummary.summary(model, (3, 32, 32))


