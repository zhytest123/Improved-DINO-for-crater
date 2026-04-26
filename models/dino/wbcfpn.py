import torch
import torch.nn as nn
import torch.nn.functional as F

class WBCFPN(nn.Module):
    """
    自定义的Weighted Bidirectional Channel FPN
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(WBCFPN, self).__init__()
        
        # print(f"WBCFPN 初始化 - 输入通道: {in_channels_list}, 输出通道: {out_channels}")
        
        # 存储原始通道列表，用于特征映射
        self.in_channels_list = in_channels_list
        
        # 上采样卷积层 - 用于从高层到低层的上采样
        self.up_sample_conv = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            self.up_sample_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 上下融合卷积层 - 将每个层级的特征投影到统一维度
        # 注意：这里按照从高层到低层的顺序创建，即从大通道数到小通道数
        self.upbottom_conv = nn.ModuleList()
        reversed_channels = in_channels_list[::-1]  # 反转通道列表
        for in_channels in reversed_channels:
            self.upbottom_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 权重生成卷积层 - 为特征融合生成权重
        self.weight_conv = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.weight_conv.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid()  # 输出0-1之间的权重
                )
            )
        
        # 侧向连接卷积层 - 最终输出处理
        self.lateral_conv = nn.ModuleList()
        for _ in in_channels_list:
            self.lateral_conv.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, num_feature_levels=None):
        """
        Args:
            features: List of feature maps from backbone [P2, P3, P4] or [P3, P4, P5]
                     Shape: [(B, C_i, H_i, W_i), ...] where C_i varies
        Returns:
            List of enhanced feature maps with uniform channels
        """
        # 反转特征列表，从高层到低层 [P5/P4, P4/P3, P3/P2]
        feature_maps = features[::-1]
        
        # 调试信息
        # print(f"输入特征形状: {[f.shape for f in feature_maps]}")
        # print(f"通道映射: {[f.shape[1] for f in feature_maps]} -> {[ch for ch in self.in_channels_list[::-1]]}")
        
        up_sample_features = []
        up_bottom_features = []
        
        # 处理最高层特征 (P5 或 P4)
        highest_feature = self.upbottom_conv[0](feature_maps[0])  # 转换到统一维度
        up_bottom_features.append(highest_feature)
        up_sample_features.append(highest_feature)
        
        # 生成上采样特征序列
        for up_sample_layer in self.up_sample_conv:
            upsampled = up_sample_layer(up_sample_features[0])
            up_sample_features.insert(0, upsampled)
        
        # 重新排序上采样特征
        up_sample_features = up_sample_features[::-1]
        
        # 融合不同层级的特征
        for i, (feature, conv, weight_conv) in enumerate(zip(
            feature_maps[1:], self.upbottom_conv[1:], self.weight_conv
        )):
            # 当前层特征投影
            down_feature = conv(feature)
            _, _, h, w = feature.shape
            
            # 获取上层特征
            high_feature = up_sample_features[i + 1]
            
            # 确保特征尺寸匹配
            if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
                high_feature = F.interpolate(
                    high_feature, size=(h, w), mode='bilinear', align_corners=False
                )
            
            # 生成自适应权重并加权融合
            attention_weights = weight_conv(high_feature)
            weighted_down_feature = attention_weights * down_feature
            fusion_feature = weighted_down_feature + high_feature
            
            up_bottom_features.append(fusion_feature)
        
        # 输出处理：从低层到高层排序
        results = up_bottom_features[::-1]
        
        # 应用最终的卷积层
        final_results = []
        for i, result in enumerate(results):
            processed = self.lateral_conv[i](result)
            final_results.append(processed)
        
        if num_feature_levels is not None and len(final_results) < num_feature_levels:
            last = final_results[-1]
            for _ in range(num_feature_levels - len(final_results)):
                # 3x3卷积+stride=2下采样
                last = nn.Conv2d(last.shape[1], last.shape[1], kernel_size=3, stride=2, padding=1).to(last.device)(last)
                last = nn.ReLU(inplace=True)(last)
                final_results.append(last)

        return final_results