# ------------------------------------------------------------------------
# HS-FPN Implementation for DINO
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP (High Frequency Perception Module)
# │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   └── DctChannelInteraction (Channel Path of HFP)
# └── SDP&SDP_Large (Spatial Dependency Perception Module
#-----------------------------------------------------------------#

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_dct as DCT
    DCT_AVAILABLE = True
except ImportError:
    print("Warning: torch_dct not available. HS-FPN will use alternative implementations.")
    DCT_AVAILABLE = False

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    print("Warning: einops not available. Using manual tensor reshaping.")
    EINOPS_AVAILABLE = False

__all__ = ['HSFPN']

def safe_manual_rearrange_to_patches(x, p1, p2):
    """Safe manual implementation of einops rearrange for patch extraction"""
    b, c, h, w = x.shape
    
    # Ensure patch sizes are reasonable
    p1 = max(1, min(p1, h))
    p2 = max(1, min(p2, w))
    
    # If patch size is too large, use adaptive pooling
    if p1 * p2 >= h * w:
        # Use global average pooling
        pooled = F.adaptive_avg_pool2d(x, (1, 1))
        return pooled.view(b, c, 1)
    
    # Ensure dimensions are divisible by patch sizes
    h_new = (h // p1) * p1
    w_new = (w // p2) * p2
    
    if h_new != h or w_new != w:
        x = x[:, :, :h_new, :w_new]
        h, w = h_new, w_new
    
    h_patches = h // p1
    w_patches = w // p2
    
    try:
        # Reshape to extract patches
        x = x.view(b, c, h_patches, p1, w_patches, p2)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (b, h_patches, w_patches, c, p1, p2)
        x = x.contiguous().view(b * h_patches * w_patches, c, p1 * p2)
        return x
    except RuntimeError as e:
        print(f"Patch extraction failed: {e}")
        # Fallback to adaptive pooling
        pooled = F.adaptive_avg_pool2d(x, (p1, p2))
        return pooled.view(b, c, p1 * p2)

def safe_manual_rearrange_from_patches(x, b, c, h_patches, w_patches, p1, p2, target_h, target_w):
    """Safe manual implementation of einops rearrange for patch reconstruction"""
    try:
        # Verify dimensions match
        expected_size = b * h_patches * w_patches * c * p1 * p2
        actual_size = x.numel()
        
        if expected_size != actual_size:
            print(f"Size mismatch: expected {expected_size}, got {actual_size}")
            # Use interpolation as fallback
            x_reshaped = x.view(b, c, -1).mean(dim=2, keepdim=True).unsqueeze(-1)
            return F.interpolate(x_reshaped, size=(target_h, target_w), mode='nearest')
        
        x = x.view(b, h_patches, w_patches, c, p1, p2)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (b, c, h_patches, p1, w_patches, p2)
        x = x.contiguous().view(b, c, h_patches * p1, w_patches * p2)
        
        # Resize to target dimensions if needed
        if x.size(2) != target_h or x.size(3) != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return x
    except RuntimeError as e:
        print(f"Patch reconstruction failed: {e}")
        # Fallback to interpolation
        x_reshaped = x.view(b, c, -1).mean(dim=2, keepdim=True).unsqueeze(-1)
        return F.interpolate(x_reshaped, size=(target_h, target_w), mode='nearest')

#------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct and DCT_AVAILABLE  # Only use DCT if available
        
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1)
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        
        # Use DCT if available
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, 1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

#------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct and DCT_AVAILABLE
        
        # Ensure groups divides in_channels
        groups = min(32, in_channels)
        while in_channels % groups != 0:
            groups -= 1
        
        self.channel1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.channel2x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        # Use DCT if available
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, 1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight  

#------------------------------------------------------------------#
# High Frequency Perception Module HFP
#------------------------------------------------------------------#
class HFP(nn.Module):
    def __init__(self, in_channels, ratio, patch=(8,8), isdct=True):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        
        # Ensure groups divides in_channels
        groups = min(32, in_channels)
        while in_channels % groups != 0:
            groups -= 1
            
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, in_channels)
        )
        
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)

#------------------------------------------------------------------#
# Simplified Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP(nn.Module):
    def __init__(self, dim=256, inter_dim=None):
        super(SDP, self).__init__()
        self.inter_dim = inter_dim
        if self.inter_dim is None:
            self.inter_dim = dim
            
        # Ensure groups divides dim
        groups = min(32, dim)
        while dim % groups != 0:
            groups -= 1
            
        self.conv_q = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), 
            nn.GroupNorm(groups, self.inter_dim)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), 
            nn.GroupNorm(groups, self.inter_dim)
        )
        
        # Simplified attention mechanism
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_dim // 4, self.inter_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_low, x_high, patch_size=None):
        b_, _, h_, w_ = x_low.size()
        
        # Use simplified global attention instead of patch-based attention
        q_feat = self.conv_q(x_low)
        k_feat = self.conv_k(x_high)
        
        # Resize k_feat to match q_feat if needed
        if k_feat.shape[2:] != q_feat.shape[2:]:
            k_feat = F.interpolate(k_feat, size=(h_, w_), mode='bilinear', align_corners=False)
        
        # Global channel attention
        q_global = self.global_pool(q_feat)
        k_global = self.global_pool(k_feat)
        
        # Compute attention weights
        combined = q_global + k_global
        attn_weights = self.channel_attn(combined)
        
        # Apply attention
        enhanced = q_feat + attn_weights * k_feat
        
        return enhanced + x_low

#------------------------------------------------------------------#
# Improved Version of Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP_Improved(nn.Module):
    def __init__(self, dim=256, inter_dim=None):
        super(SDP_Improved, self).__init__()
        self.inter_dim = inter_dim
        if self.inter_dim is None:
            self.inter_dim = dim
            
        # Ensure groups divides dim
        groups = min(32, dim)
        while dim % groups != 0:
            groups -= 1
            
        self.conv_q = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), 
            nn.GroupNorm(groups, self.inter_dim)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), 
            nn.GroupNorm(groups, self.inter_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.inter_dim, dim, 3, padding=1, bias=False), 
            nn.GroupNorm(groups, dim)
        )
        
        # Simplified attention mechanism
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_dim // 4, self.inter_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_low, x_high, patch_size=None):
        b_, _, h_, w_ = x_low.size()
        
        # Use simplified global attention
        q_feat = self.conv_q(x_low)
        k_feat = self.conv_k(x_high)
        
        # Resize k_feat to match q_feat if needed
        if k_feat.shape[2:] != q_feat.shape[2:]:
            k_feat = F.interpolate(k_feat, size=(h_, w_), mode='bilinear', align_corners=False)
        
        # Global channel attention
        q_global = self.global_pool(q_feat)
        k_global = self.global_pool(k_feat)
        
        # Compute attention weights
        combined = q_global + k_global
        attn_weights = self.channel_attn(combined)
        
        # Apply attention and additional conv
        enhanced = q_feat + attn_weights * k_feat
        output = self.conv(enhanced + x_low)
        
        return output

#------------------------------------------------------------------#
# HS_FPN - Adapted for DINO
#------------------------------------------------------------------#
class HSFPN(nn.Module):
    """
    HS-FPN implementation adapted for DINO framework
    """
    def __init__(self, in_channels_list, out_channels=256, ratio=(0.25, 0.25)):
        super(HSFPN, self).__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_ins = len(in_channels_list)
        
        # Lateral convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Extra convolutions for additional feature levels
        self.extra_convs = nn.ModuleList()
        for _ in range(2):  # Support up to 2 additional levels beyond backbone
            self.extra_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Upsample function
        def interpolate(input_tensor):
            return F.interpolate(input_tensor, scale_factor=2, mode='nearest')
        self.fpn_upsample = interpolate 
        
        # HFP modules for different levels
        self.SelfAttn_p4 = HFP(out_channels, ratio=None, isdct=False)
        self.SelfAttn_p3 = HFP(out_channels, ratio=None, isdct=False)  
        self.SelfAttn_p2 = HFP(out_channels, ratio=ratio, patch=(8, 8), isdct=True)
        self.SelfAttn_p1 = HFP(out_channels, ratio=ratio, patch=(16, 16), isdct=True)

        # SDP modules for cross-level interactions
        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)
        
    def forward(self, inputs, num_feature_levels=None):
        """
        Args:
            inputs: List of feature maps from backbone
            num_feature_levels: Number of feature levels (for compatibility)
        Returns:
            List of enhanced feature maps
        """
        assert len(inputs) == len(self.in_channels_list)
        
        # Apply lateral convolutions
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # Apply HS-FPN processing
        # Features are ordered from high-res to low-res: [P2, P3, P4, P5, ...]
        num_levels = len(laterals)
        
        if num_levels >= 4:
            # Process from lowest resolution (highest index) to highest resolution (index 0)
            # Apply self-attention to the lowest resolution level
            laterals[-1] = self.SelfAttn_p4(laterals[-1])
            
            # For 4+ levels, apply the standard HS-FPN cross-attention pattern
            if num_levels >= 4:
                # Process P4 (second lowest resolution)
                laterals[-2] = self.CrossAtten_p4_p3(
                    self.SelfAttn_p3(laterals[-2]), 
                    self.fpn_upsample(laterals[-1])
                )
                
                # Process P3 (third lowest resolution)  
                laterals[-3] = self.CrossAtten_p3_p2(
                    self.SelfAttn_p2(laterals[-3]), 
                    self.fpn_upsample(laterals[-2])
                )
                
                # Process P2 (fourth lowest resolution)
                laterals[-4] = self.CrossAtten_p2_p1(
                    self.SelfAttn_p1(laterals[-4]), 
                    self.fpn_upsample(laterals[-3])
                )
                
                # Handle additional levels (P1, P0, etc.) if we have 5+ levels
                for i in range(num_levels - 5, -1, -1):
                    # For additional higher resolution levels, use P1 processing
                    laterals[i] = self.CrossAtten_p2_p1(
                        self.SelfAttn_p1(laterals[i]),
                        self.fpn_upsample(laterals[i + 1])
                    )
        else:
            # Handle fewer than 4 levels
            for i in range(len(laterals)):
                if i == len(laterals) - 1:  # Lowest resolution
                    laterals[i] = self.SelfAttn_p4(laterals[i])
                elif i == len(laterals) - 2:  # Second lowest resolution
                    laterals[i] = self.SelfAttn_p3(laterals[i])
                else:  # Higher resolution levels
                    laterals[i] = self.SelfAttn_p2(laterals[i])
        
        # Standard FPN top-down processing (from low-res to high-res)
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # Apply final convolutions
        outs = []
        for i in range(len(laterals)):
            outs.append(self.fpn_convs[i](laterals[i]))
        
        # Handle additional feature levels if needed
        if num_feature_levels is not None and len(outs) < num_feature_levels:
            last = outs[-1]
            extra_idx = 0
            for _ in range(num_feature_levels - len(outs)):
                if extra_idx < len(self.extra_convs):
                    last = self.extra_convs[extra_idx](last)
                    extra_idx += 1
                else:
                    # Fallback: simple max pooling with stride 2
                    last = F.max_pool2d(last, kernel_size=2, stride=2)
                outs.append(last)
        
        return outs