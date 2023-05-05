# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext import LayerNorm
from models.convnext import Block as ConvNeXtBlock


# 2d conv blocks
class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes: 
            self.skip_con = nn.Sequential() 
        else:
            self.skip_con = nn.Sequential(  # adjust the dimension
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)  # skip connection
        return F.relu(res + skip, True)


class Pool2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes) -> None:
        super().__init__()
        self.downsample_layer = nn.Sequential(
                    LayerNorm(in_planes, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=2),
            )
        
    def forward(self, x):
        return self.downsample_layer(x)
    

class Upsample2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecoder(nn.Module):
    def __init__(self, dims=[48, 96, 192]):
        super(EncoderDecoder, self).__init__()

        self.encoder_pool1 = Pool2DBlock(dims[0], dims[1])
        self.encoder_res1 = ConvNeXtBlock(dims[1])
        self.encoder_pool2 = Pool2DBlock(dims[1], dims[2])
        self.encoder_res2 = ConvNeXtBlock(dims[2])

        self.mid_res = ConvNeXtBlock(dims[2])

        self.decoder_res2 = ConvNeXtBlock(dims[2])
        self.decoder_upsample2 = Upsample2DBlock(dims[2], dims[1], 2, 2)
        self.decoder_res1 = ConvNeXtBlock(dims[1])
        self.decoder_upsample1 = Upsample2DBlock(dims[1], dims[0], 2, 2)

        self.skip_res1 = ConvNeXtBlock(dims[0])
        self.skip_res2 = ConvNeXtBlock(dims[1])

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class P2PNet(nn.Module):
    def __init__(self, input_channels, output_channels, dims=[48, 96, 192]):
        super(P2PNet, self).__init__()
        self.output_channels = output_channels

        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, dims[0]//2, 7),
            Res2DBlock(dims[0]//2, dims[0]),
        )

        self.encoder_decoder = EncoderDecoder(dims=[48, 96, 192])

        self.output_layer = nn.Conv2d(dims[0], output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class P2PNeXt(nn.Module):
    def __init__(self, channels=15, depths=[1, 1, 3, 1], dims=[48, 96, 192, 384], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.encoder_decoder = EncoderDecoderConvNeXt(channels, 
                                                      depths=depths,
                                                      dims=dims,
                                                      drop_path_rate=drop_path_rate,
                                                      layer_scale_init_value=layer_scale_init_value)
        self.output_layer = nn.Sequential(
            ConvNeXtBlock(dims[0], layer_scale_init_value=layer_scale_init_value),
            nn.Conv2d(dims[0], channels, kernel_size=1)
        )
    

    def forward(self, x):
        x = self.encoder_decoder(x)
        x = self.output_layer(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class CenterNet(nn.Module):
    def __init__(self, input_channels, output_channels, head_conv=48, dims=[48, 96, 192]):
        super(CenterNet, self).__init__()
        self.output_channels = output_channels

        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, dims[0]//2, 7),
            Res2DBlock(dims[0]//2, dims[0]),
        )

        self.encoder_decoder = EncoderDecoder()

        self.output_hm = nn.Sequential(
            nn.Conv2d(dims[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, output_channels, kernel_size=1, padding=0)
        )

        self.output_size = nn.Sequential(
            nn.Conv2d(dims[0], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, padding=0, bias=True)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x, _ = torch.max(x, dim=4) # max-pooling along z-axis
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        hm, size = self.output_hm(x), self.output_size(x)
        return hm, size

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class CenterNeXt(nn.Module):
    def __init__(self, input_channels=15, depths=[1, 1, 3, 1], dims=[48, 96, 192, 384], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.encoder_decoder = EncoderDecoderConvNeXt(input_channels, 
                                                      depths=depths,
                                                      dims=dims,
                                                      drop_path_rate=drop_path_rate,
                                                      layer_scale_init_value=layer_scale_init_value)
        self.output_hm = nn.Sequential(
            ConvNeXtBlock(dims[0], layer_scale_init_value=layer_scale_init_value),
            nn.Conv2d(dims[0], 1, kernel_size=1)
        )
        self.output_size = nn.Sequential(
            ConvNeXtBlock(dims[0], layer_scale_init_value=layer_scale_init_value),
            nn.Conv2d(dims[0], 2, kernel_size=1)
        )
        

    def forward(self, x):
        x, _ = torch.max(x, dim=4) # max-pooling along z-axis
        x = self.encoder_decoder(x)
        hm, size = self.output_hm(x), self.output_size(x)
        return hm, size
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class EncoderDecoderConvNeXt(nn.Module):
    def __init__(self, in_chans, depths=[1, 1, 3, 1], dims=[64, 128, 256, 512], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        # downsample layers in encoder
        self.encoder_downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=1, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.encoder_downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.encoder_downsample_layers.append(downsample_layer)

        # convolution layers in encoder
        self.encoder_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + 3)] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], 
                                drop_path=dp_rates[cur + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.encoder_stages.append(stage)
            cur += depths[i]

        # upsample layers in decoder
        self.decoder_upsample_layers = nn.ModuleList() # 3 upsampling deconv layers
        self.decoder_conv_cat = nn.ModuleList()
        for i in range(3):
            upsample_layer = nn.Sequential(
                    LayerNorm(dims[-(i+1)], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(dims[-(i+1)], dims[-(i+2)], kernel_size=4, stride=2, padding = 1),
            )
            self.decoder_upsample_layers.append(upsample_layer)
            self.decoder_conv_cat.append(nn.Conv2d(dims[-(i+2)] * 2, dims[-(i+2)], kernel_size=1))
        
        # convolution layers in decoder
        self.decoder_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(3):
            stage = ConvNeXtBlock(dim=dims[-(i+2)], 
                                  drop_path=dp_rates[cur], 
                                  layer_scale_init_value=layer_scale_init_value)
            self.decoder_stages.append(stage)
            cur += 1

        # skip-connection layers
        self.skip_connections = nn.ModuleList()
        for i in range(3):
            self.skip_connections.append(nn.Conv2d(dims[i], dims[i], kernel_size=1))


    def forward(self, x):
        # encoder
        skip = []
        for i in range(4):
            x = self.encoder_downsample_layers[i](x)
            x = self.encoder_stages[i](x)
            if i < 3:
                skip.append(self.skip_connections[i](x))
        
        # decoder
        for i in range(3):
            x = self.decoder_upsample_layers[i](x)
            x = torch.cat([x, skip[2-i]], dim=1)
            x = self.decoder_conv_cat[i](x)
            x = self.decoder_stages[i](x)
        
        return x