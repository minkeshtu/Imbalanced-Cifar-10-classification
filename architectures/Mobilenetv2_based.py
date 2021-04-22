import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
if __name__ == '__main__':
    from Mobilenetv2_core import Mobilenetv2_base, Mobilenetv2_linear_bottleneck_block
else:
    from architectures.Mobilenetv2_core import Mobilenetv2_base, Mobilenetv2_linear_bottleneck_block
import torchsummary


class ConvTranspose2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(ConvTranspose2d_block, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        model = Mobilenetv2_base()
        self.encoder = nn.Sequential(*list(model.children()))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.t_conv1 = ConvTranspose2d_block(320, 160, kernel_size=2, stride=2)
        self.linear_bottleneck1 = Mobilenetv2_linear_bottleneck_block(in_planes=160, out_planes=160, stride=1, expansion=3)
        self.t_conv2 = ConvTranspose2d_block(160, 96, kernel_size=2, stride=2)
        self.linear_bottleneck2 = Mobilenetv2_linear_bottleneck_block(in_planes=96, out_planes=96, stride=1, expansion=3)
        self.t_conv3 = ConvTranspose2d_block(96, 64, kernel_size=2, stride=2)
        self.linear_bottleneck3 = Mobilenetv2_linear_bottleneck_block(in_planes=64, out_planes=64, stride=1, expansion=3)
        self.Conv1 = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.linear_bottleneck1(x)
        x = self.t_conv2(x)
        x = self.linear_bottleneck2(x)
        x = self.t_conv3(x)
        x = self.linear_bottleneck3(x)
        x = self.Conv1(x)
        x = self.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dense_size=1280):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(320, dense_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(dense_size)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(dense_size, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class AE_model(nn.Module):
    def __init__(self):
        super(AE_model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class End_to_End_model(nn.Module):
    def __init__(self, n_classes=10, dense_size= 1280):
        super(End_to_End_model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier(n_classes, dense_size)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        classifier_out = self.classifier(encoder_out)
        return decoder_out, classifier_out


class Classification_model(nn.Module):
    def __init__(self, n_classes=10, dense_size= 1280):
        super(Classification_model, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier(n_classes, dense_size)

    def forward(self, x):
        encoder_out = self.encoder(x)
        classifier_out = self.classifier(encoder_out)
        return classifier_out


class Encoder_for_visualization(nn.Module):
    def __init__(self):
        super(Encoder_for_visualization, self).__init__()
        self.vector_size = 320
        self.encoder = Encoder()
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        encoder_out = self.encoder(x)
        feature_vector = self.gap(encoder_out)
        feature_vector = feature_vector.view(-1, self.vector_size)
        return feature_vector


if __name__ == "__main__":
    model = End_to_End_model(n_classes=10, dense_size=1280)
    #model = AE_model()
    #model = Classification_model(n_classes=10, dense_size=1280)
    #print(f'# of Parameters: {sum(p.numel() for p in model.parameters())/1e6 :.2f} M' )

    import torchsummary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torchsummary.summary(model, (3, 32, 32), col_names=["input_size", "output_size"], depth=4)

