import torch
import torch.nn as nn
import torch.nn.functional as F

class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures"""
    def __init__(self, in_channels, out_channels, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.encoder = nn.Conv2d(in_channels // 4, 
                                self.up_factor ** 2 * self.kernel_size ** 2,
                                self.kernel_size, padding=self.kernel_size // 2)
        self.out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        
        # Kernel prediction
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, 
                                            self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # Content-aware reassembly
        feat = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2)
        feat = feat.view(N, C, self.kernel_size ** 2, H, W).permute(0, 3, 4, 1, 2)
        
        out = torch.matmul(feat, kernel_tensor)  # (N, H, W, C, S^2)
        out = out.permute(0, 3, 4, 1, 2).view(N, C, self.up_factor, self.up_factor, H, W)
        out = out.permute(0, 1, 4, 2, 5, 3).reshape(N, C, H * self.up_factor, 
                                                    W * self.up_factor)
        out = self.out(out)
        return out

class CARAFEPack(nn.Module):
    """CARAFE Pack for FPN"""
    def __init__(self, in_channels_list, out_channels=256):
        super(CARAFEPack, self).__init__()
        self.carafe_up = nn.ModuleList([
            CARAFE(in_channels, out_channels) 
            for in_channels in in_channels_list[:-1]
        ])
        self.carafe_down = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])

    def forward(self, inputs):
        # Lateral connections
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down with CARAFE
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self.carafe_up[i - 1](laterals[i])
        
        return tuple(laterals)