import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
torch.backends.cudnn.enabled = False


class DeformableResBlock(nn.Module):
    def __init__(self, channels):
        super(DeformableResBlock, self).__init__()

        self.offset_conv = nn.Conv2d(channels, 2*3*3, 3, 1, 1)
        self.deform_conv = DeformConv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(False)
        
    def forward(self, x):
        identity = x

        offset = self.offset_conv(x)

        out = self.deform_conv(x, offset)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out



class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=False)
        )
    
    def forward(self, x):

        return self.layers(x)



class DoubleConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=False),
        )
    
    def forward(self, x):

        return self.layers(x)



class ConvBNReluUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBNReluUp, self).__init__()
        self.conv = ConvBNRelu(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        
        return self.upsample(self.conv(x))



class ConvBNReluUpIn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReluUpIn, self).__init__()
        self.conv = ConvBNRelu(in_channels, out_channels)

    def forward(self, x):
        
        return F.interpolate(self.conv(x), scale_factor=2)



class ConvBNReluDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReluDown, self).__init__()
        self.conv = ConvBNRelu(in_channels, out_channels)

    def forward(self, x):
        
        return F.interpolate(self.conv(x), scale_factor=0.5)



class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, channels_out, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(channels_out)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride, bias=False),
                nn.BatchNorm2d(channels_out)
            )

    def forward(self, imgs):
        out = self.conv(imgs)
        out += self.shortcut(imgs)
        out = F.relu(out)
        return out



class BottleneckBlock(nn.Module):
	"""
	A Bottleneck Block
	"""
	def __init__(self, in_channels, out_channels, r, drop_rate=1, groups=1, dilation=1, se=True):
		super(BottleneckBlock, self).__init__()

		self.downsample = None
		self.se = se

		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(False)

		if (drop_rate == 2) or (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, groups=groups, dilation=dilation, bias=False),
				Normalize(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					    stride=drop_rate, groups=groups, dilation=dilation, padding=0, bias=False),
			Normalize(out_channels),
			Activation,
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, 
						groups=groups, dilation=dilation, bias=False),
			Normalize(out_channels),
			Activation,
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, 
						groups=groups, dilation=dilation, bias=False),
			Normalize(out_channels)
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)

		if self.se:
			scale = self.se(x)
			x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)

		return x



class SENet(nn.Module):
	"""
	SENet with BottleneckBlock
	"""
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, groups=1, dilation=1, drop_rate=1, se=True):
		super(SENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)



class SENet_decoder(nn.Module):
	"""
	SENet with BottleneckBlock
	"""
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, groups=1, dilation=1, drop_rate=1, drop_rate2=1, se=True):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r=r, drop_rate=1, se=se, groups=groups, dilation=dilation)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r=r, drop_rate=1, se=se, groups=groups, dilation=dilation)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate2, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)
			out_channels *= drop_rate2
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)