from .layers import *

# 定义反转块
class ReverserBlock(nn.Module):
    def __init__(self, channels):
        super(ReverserBlock, self).__init__()

        self.channels = channels

        self.layers = nn.Sequential(
            DeformableResBlock(self.channels),
            SENet(self.channels, self.channels, blocks=4),
            SENet_decoder(self.channels, self.channels, blocks=2),
            nn.ReLU(False),
            SENet(self.channels, self.channels, blocks=4),
            SENet_decoder(self.channels, self.channels, blocks=2),
            nn.ReLU(False),
        )

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = ConvBNRelu(channels, channels, kernel_size=3, stride=1, padding=1)
        
         
    def forward(self, x):
        return self.conv(self.downsample(self.layers(x)))


# 定义消息反转器（MR）
class DeformableDecoder(nn.Module):
    def __init__(self, message_length=64, channels=64, N2=3):
        super(DeformableDecoder, self).__init__()

        self.conv_first = DoubleConvBNRelu(3, channels)

        reverse_blocks = [ReverserBlock(channels)] if N2 != 0 else []

        for _ in range(N2 - 1):
            layer=ReverserBlock(channels)
            reverse_blocks.append(layer)
        self.reverse_blocks = nn.Sequential(*reverse_blocks)

        self.conv_last = DoubleConvBNRelu(channels, 1)

        self.linear1 = nn.Linear(32*32, 256)

        self.linear2 = nn.Linear(256, message_length)
        
    def forward(self, Pd):
        features = []

        # x shape: [B, 3, 512, 512]
        x = self.conv_first(Pd)
        features.append(x)
        Ir = self.reverse_blocks(x)  # x shape: [B, 40, 32, 32]
        features.append(Ir)
        Ir = self.conv_last(Ir)  # x shape: [B, 1, 32, 32]
        features.append(Ir)
        Ir = Ir.view(Ir.size(0), -1)  # x shape: [B, 32 * 32]
        Ir = self.linear1(Ir)  # x shape: [B, 256]
        Ir = self.linear2(Ir)  # x shape: [B, message_length]
        return Ir, features