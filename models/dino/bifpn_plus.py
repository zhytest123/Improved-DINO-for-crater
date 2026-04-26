import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPNPlus(nn.Module):
    """Enhanced BiFPN with attention and dynamic weights"""
    def __init__(self, in_channels_list, out_channels=256, num_outs=5, epsilon=1e-4):
        super(BiFPNPlus, self).__init__()
        self.epsilon = epsilon
        self.num_outs = num_outs
        
        # Lateral convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        # BiFPN weights (learnable)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        
        # Attention mechanism
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels//4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels//4, out_channels, 1),
                nn.Sigmoid()
            ) for _ in range(len(in_channels_list))
        ])
        
        # Dynamic weight generation
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//8, len(in_channels_list), 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        Args:
            inputs: List of feature maps [P2, P3, P4, P5]
        Returns:
            tuple: Enhanced multi-scale features
        """
        # Lateral connections
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        
        # Bottom-up pathway with weighted fusion
        outs = []
        for i in range(used_backbone_levels):
            if i == 0:
                outs.append(laterals[i])
            else:
                # Weighted fusion
                w = F.relu(self.w1)
                w = w / (torch.sum(w, dim=0) + self.epsilon)
                
                prev_shape = laterals[i].shape[2:]
                fused = w[0] * laterals[i] + w[1] * F.interpolate(
                    outs[i-1], size=prev_shape, mode='nearest')
                outs.append(fused)
        
        # Apply attention and final convolution
        final_outs = []
        for i, out in enumerate(outs):
            att = self.attention[i](out)
            out = out * att
            out = self.fpn_convs[i](out)
            final_outs.append(out)
        
        return tuple(final_outs)