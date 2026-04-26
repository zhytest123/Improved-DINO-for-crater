import torch
import torch.nn as nn
import torch.nn.functional as F

class ASFF(nn.Module):
    """Adaptively Spatial Feature Fusion"""
    def __init__(self, level, in_channels_list, out_channels=256):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128] # Assuming 3 levels
        self.inter_dim = out_channels
        
        # Compression convs
        if level == 0:
            self.stride_level_1 = nn.Conv2d(in_channels_list[1], self.inter_dim, 3, 2, 1)
            self.stride_level_2 = nn.Conv2d(in_channels_list[2], self.inter_dim, 3, 4, 1)
            self.expand = nn.Conv2d(self.inter_dim, out_channels, 3, 1, 1)
        elif level == 1:
            self.compress_level_0 = nn.Conv2d(in_channels_list[0], self.inter_dim, 1, 1, 0)
            self.stride_level_2 = nn.Conv2d(in_channels_list[2], self.inter_dim, 3, 2, 1)
            self.expand = nn.Conv2d(self.inter_dim, out_channels, 3, 1, 1)
        elif level == 2:
            self.compress_level_0 = nn.Conv2d(in_channels_list[0], self.inter_dim, 1, 1, 0)
            self.expand = nn.Conv2d(self.inter_dim, out_channels, 3, 1, 1)

        # Weight generation networks
        compress_c = 8
        self.weight_level_0 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_1 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, 
                                          size=x_level_1.shape[2:], mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed,
                                          size=x_level_2.shape[2:], mode='nearest')
            level_1_resized = F.interpolate(x_level_1,
                                          size=x_level_2.shape[2:], mode='nearest')
            level_2_resized = x_level_2

        # Generate adaptive weights
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # Weighted fusion
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                           level_1_resized * levels_weight[:, 1:2, :, :] + \
                           level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)
        return out