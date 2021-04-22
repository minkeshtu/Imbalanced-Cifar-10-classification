import torch
import torchsummary
import torch.nn as nn

class DoubleConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConv2d_block, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


class ConvTranspose2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvTranspose2d_block, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, n_channels, filters):
        super(Encoder, self).__init__()
        self.encode_conv1 = DoubleConv2d_block(n_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encode_conv2 = DoubleConv2d_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encode_conv3 = DoubleConv2d_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encode_conv4 = DoubleConv2d_block(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.encode_conv1(x)
        x = self.pool1(x)
        x = self.encode_conv2(x)
        x = self.pool2(x)
        x = self.encode_conv3(x)
        x = self.pool3(x)
        x = self.encode_conv4(x)
        x = self.pool4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_channels, filters):
        super(Decoder, self).__init__()
        self.t_conv1 = ConvTranspose2d_block(filters[3], filters[2])
        self.decode_conv1 = DoubleConv2d_block(filters[2], filters[2])
        self.t_conv2 = ConvTranspose2d_block(filters[2], filters[1])
        self.decode_conv2 = DoubleConv2d_block(filters[1], filters[1])
        self.t_conv3 = ConvTranspose2d_block(filters[1], filters[0])
        self.decode_conv3 = DoubleConv2d_block(filters[0], filters[0])
        self.t_conv4 = ConvTranspose2d_block(filters[0], n_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.decode_conv1(x)
        x = self.t_conv2(x)
        x = self.decode_conv2(x)
        x = self.t_conv3(x)
        x = self.decode_conv3(x)
        x = self.t_conv4(x)
        x = self.sigmoid(x)
        return x


class Dense_block(nn.Module):
    def __init__(self, in_channels, dense_size, drop_rate):
        super(Dense_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, dense_size),
            nn.ReLU(inplace=True)
            #nn.Dropout(drop_rate)
        )
    def forward(self, x):
        return self.layers(x)
        

class Classifier(nn.Module):
    def __init__(self, n_classes, in_channels, dense_size, drop_rate):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dense_block = Dense_block(in_channels, dense_size, drop_rate)
        self.out = nn.Linear(dense_size, n_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(-1, self.in_channels)
        x = self.dense_block(x)
        x = self.out(x)
        return x


class AE_model(nn.Module):
    def __init__(self, n_channels=3, filters = [64, 128, 256, 512]):
        super(AE_model, self).__init__()
        self.encoder = Encoder(n_channels, filters)
        self.decoder = Decoder(n_channels, filters)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


class End_to_End_model(nn.Module):
    def __init__(self, n_channels=3, n_classes=10, filters = [64, 128, 256, 512], dense_size = 512, drop_rate=0.5):
        super(End_to_End_model, self).__init__()
        self.encoder = Encoder(n_channels, filters)
        self.decoder = Decoder(n_channels, filters)
        self.classifier = Classifier(n_classes, filters[3], dense_size, drop_rate)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        classifier_out = self.classifier(encoder_out)
        return decoder_out, classifier_out


class Classification_model(nn.Module):
    def __init__(self, n_channels=3, n_classes=10, filters = [64, 128, 256, 512], dense_size = 512, drop_rate=0.5):
        super(Classification_model, self).__init__()
        self.encoder = Encoder(n_channels, filters)
        self.classifier = Classifier(n_classes, filters[3], dense_size, drop_rate)

    def forward(self, x):
        encoder_out = self.encoder(x)
        classifier_out = self.classifier(encoder_out)
        return classifier_out


class Encoder_for_visualization(nn.Module):
    def __init__(self, n_channels=3, filters = [64, 128, 256, 512]):
        super(Encoder_for_visualization, self).__init__()
        self.vector_size = filters[-1]
        self.encoder = Encoder(n_channels, filters)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        encoder_out = self.encoder(x)
        feature_vector = self.gap(encoder_out)
        feature_vector = feature_vector.view(-1, self.vector_size)
        return feature_vector


if __name__ == "__main__":
    width_scale_factor = 2
    filters = [int(width_scale_factor*i) for i in [32, 64, 128, 256]]
    
    #model = Classification_model(3, 10, filters=filters, dense_size=512)
    #model = AE_model(n_channels=3, filters=filters)
    model = End_to_End_model(3, 10, filters=filters, dense_size=512)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torchsummary.summary(model, (3, 32, 32), branching=True, col_names=["input_size", "output_size"], depth=5)