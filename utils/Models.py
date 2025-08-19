"""
Optimized segmentation models:
- Pre-trained on L3 gastric data (~200 patients).
- Lightweight architecture & fast when compiled.

"""
import torch
import torch.nn as nn
from torchvision.models.shufflenetv2 import ShuffleNetV2, InvertedResidual

import segmentation_models_pytorch as smp

class Titan(ShuffleNetV2):
    #* ShuffleNetv2 adapted for segmentation

    def __init__(self, stages_repeats, stages_out_channels,
        num_classes=1, p_dropout=0.25):

        super(Titan, self).__init__(stages_repeats, stages_out_channels, num_classes=num_classes, inverted_residual=InvertedResidual)
        #* Encoder should be initialised now
        #! Setup decoder
        # Segmentation head
        self.segmentation_head = nn.Conv2d(in_channels=24, out_channels=num_classes, kernel_size=1)
        self.upsample0 = transpose_conv(in_channels=24, out_channels=24, p_dropout=p_dropout, compress_factor=2)

        self.upsample1 = transpose_conv(in_channels=24, out_channels=24, p_dropout=p_dropout, compress_factor=2)
        self.up_conv_1 = asym_conv(in_channels=24+24, out_channels=24, p_dropout=p_dropout)
        
        self.upsample2 = transpose_conv(in_channels=48, out_channels=48, p_dropout=p_dropout, compress_factor=2)
        self.up_conv_2 = asym_conv(in_channels=48+24, out_channels=24, p_dropout=p_dropout)
        
        self.upsample3 = transpose_conv(in_channels=96, out_channels=96, p_dropout=p_dropout, compress_factor=2)
        self.up_conv_3 = asym_conv(in_channels=96+48, out_channels=48, p_dropout=p_dropout)
        
        self.upsample4 = transpose_conv(in_channels=192, out_channels=192, p_dropout=p_dropout, compress_factor=2)
        self.up_conv_4 = asym_conv(in_channels=192+96, out_channels=96, p_dropout=p_dropout)
        

    
    def _forward_impl(self,x):
        # See note [TorchScript super()]
        #* Stage 1
        start = self.conv1(x)
        down1 = self.maxpool(start)
        #* Each stage =>

        #* InvertedResidual (in, out, stride=2) -> InvertedResidual (out, out, stride=1)
        down2 = self.stage2(down1)
        down3 = self.stage3(down2)
        down4 = self.stage4(down3)

        #* Decoder
        #4.
        x = self.upsample4(down4)
        x = torch.cat([x, down3], dim=1)
        x = self.up_conv_4(x)
        #3.
        x = self.upsample3(x)
        x = torch.cat([x, down2], dim=1)
        x = self.up_conv_3(x)
        #2
        x = self.upsample2(x)
        x = torch.cat([x, down1], dim=1)
        x = self.up_conv_2(x)
        #1.
        x = self.upsample1(x)
        x = torch.cat([x, start], dim=1)
        x = self.up_conv_1(x)
        # Back to original size
        x = self.upsample0(x)
        return self.segmentation_head(x)
        
    def forward(self, x):
        return self._forward_impl(x)

def Titan_base(**kwargs):
    return Titan([4,8,4], [24,48,96,192,512], **kwargs)


#~~~ MODULES ~~~
class transpose_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout, compress_factor=2):
        super(transpose_conv, self).__init__()
        self.intermediate_out = in_channels//compress_factor
        # Definition
        self.transpose_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_out, kernel_size=(1,1)),
            nn.BatchNorm2d(self.intermediate_out),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
            nn.ConvTranspose2d(in_channels=self.intermediate_out, out_channels=self.intermediate_out,
             kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(self.intermediate_out),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
            nn.Conv2d(in_channels=self.intermediate_out, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout)
        )
    
    def forward(self, x):
        return self.transpose_conv(x)


class asym_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout, compress_factor=2):
        super(asym_conv, self).__init__()
        self.intermediate_out = in_channels//compress_factor
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_out, kernel_size=(1,1)),
            nn.BatchNorm2d(self.intermediate_out),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
            ## Assymetric conv
            nn.Conv2d(in_channels=self.intermediate_out, out_channels=self.intermediate_out,
                    kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(in_channels=self.intermediate_out, out_channels=self.intermediate_out,
                    kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(self.intermediate_out),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
            nn.Conv2d(in_channels=self.intermediate_out, out_channels=out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
        )
        if in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        if self.residual_conv is None:
            return self.bottleneck_conv(x) + x
        else:
            return self.bottleneck_conv(x) + self.residual_conv(x)
        

class Titan_vit(smp.Unet):
    def __init__(self, inputChannels=3, num_classes=4, encoder='tu-mixnet_m'):
        print('Loading Titan Vision Transformer')
        super().__init__(encoder_name=encoder,
                    encoder_weights="imagenet",
                    in_channels=inputChannels,
                    classes=num_classes)
        # self.model = smp.Unet(
            
        # )