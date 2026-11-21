import torch 
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False), # 3 1 1
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False), # 3 1 1
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)    


class UNET3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], up_stop=0):
        super(UNET3D, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # down part
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # up part
        for feature in reversed(features[up_stop:]):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)   
            )
             
            self.ups.append(DoubleConv3D(feature * 2, feature))

        # bottleneck & final
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[up_stop], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) #ConvTranspose2d
            skip_connection = skip_connections[idx//2]

            assert x.shape == skip_connection.shape

            concat_skip = torch.cat((skip_connection, x), dim=1) # dim 1 = channel dimension (batch, channel, height, with)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
