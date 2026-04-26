# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .convnext import build_convnext
from .swin_transformer import build_swin_transformer



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices:list,
                 batch_norm=FrozenBatchNorm2d,
                 ):
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=batch_norm)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4-len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class EnhancedJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, fpn_type='wbcfpn', num_feature_levels=None):
        super().__init__(backbone, position_embedding)
        
        # 获取backbone输出通道数
        bb_num_channels = backbone.num_channels
        out_channels = 256  # FPN统一输出通道数
        self.num_feature_levels = num_feature_levels
        
        # 根据类型选择FPN模块
        if fpn_type == 'bifpn':
            from .bifpn_plus import BiFPNPlus
            self.fpn = BiFPNPlus(bb_num_channels, out_channels=out_channels)
        elif fpn_type == 'asff':
            from .asff import ASFF
            self.fpn = nn.ModuleList([
                ASFF(i, bb_num_channels, out_channels) for i in range(len(bb_num_channels))
            ])
        elif fpn_type == 'carafe':
            from .carafe import CARAFEPack
            self.fpn = CARAFEPack(bb_num_channels, out_channels)
        elif fpn_type == 'wbcfpn':
            from .wbcfpn import WBCFPN
            self.fpn = WBCFPN(bb_num_channels, out_channels)
        elif fpn_type == 'hsfpn':
            from .hsfpn import HSFPN
            self.fpn = HSFPN(bb_num_channels, out_channels)    
        else:
            self.fpn = None
            
        # 设置输出通道数（这里会被build_backbone函数覆盖，但保持一致性）
        self.output_channels = out_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # backbone
        out: List[NestedTensor] = []
        pos = []
        
        # 提取特征和mask，确保顺序正确
        features = []
        masks = []
        for name in sorted(xs.keys()):  # 确保顺序一致
            x = xs[name]
            features.append(x.tensors)
            masks.append(x.mask)
        
        # 应用FPN
        if self.fpn is not None:
            if isinstance(self.fpn, nn.ModuleList):  # ASFF
                enhanced_features = [
                    asff_layer(*features) for asff_layer in self.fpn
                ]
            else:  # BiFPN, CARAFE, WBCFPN
                enhanced_features = self.fpn(features, num_feature_levels=self.num_feature_levels)
        else:
            enhanced_features = features
        
        # 重新打包为NestedTensor
        for i, (feat, mask) in enumerate(zip(enhanced_features, masks)):
            out.append(NestedTensor(feat, mask))
            pos.append(self[1](NestedTensor(feat, mask)).to(feat.dtype))
        
        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now
        - use_saconv: whether to use SAConv to replace 3x3 conv in ResNet

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)
    use_saconv = getattr(args, 'use_saconv', False)
    use_arconv = getattr(args, 'use_arconv', False)

    if args.backbone in ['resnet50', 'resnet101']:
        if args.backbone in ['resnet50', 'resnet101']:
            if use_arconv:
                from .arconv import ARConvBackbone
                backbone = ARConvBackbone(args.backbone, train_backbone, args.dilation,
                                        return_interm_indices,
                                        batch_norm_type=getattr(args, 'batch_norm_type', 'FrozenBatchNorm2d'),
                                        arconv_mode=getattr(args,'arconv_mode','layer4_only'),
                                        arconv_verbose=getattr(args, 'arconv_verbose', False),
                                        arconv_only_stride1=getattr(args, 'arconv_only_stride1', True))

            elif use_saconv:
                # 使用SAConv版本的ResNet
                from .saconv import SAConvBackbone
                backbone = SAConvBackbone(args.backbone, train_backbone, args.dilation,   
                                        return_interm_indices,   
                                        batch_norm_type=getattr(args, 'batch_norm_type', 'FrozenBatchNorm2d'))
            else:
                # 使用原始ResNet
                backbone = Backbone(args.backbone, train_backbone, args.dilation,   
                                return_interm_indices,   
                                batch_norm=FrozenBatchNorm2d)
        bb_num_channels = backbone.num_channels
    elif args.backbone in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, \
                    pretrain_img_size=pretrain_img_size, \
                    out_indices=tuple(return_interm_indices), \
                dilation=args.dilation, use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break
        if "backbone_dir" in args:
            pretrained_dir = args.backbone_dir
            PTDICT = {
                'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
                'swin_B_384_22k': 'swin_base_patch4_window12_384.pth',
                'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
            }
            pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            def key_select_function(keyname):
                if 'head' in keyname:
                    return False
                if args.dilation and 'layers.3' in keyname:
                    return False
                return True
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
    elif args.backbone in ['convnext_xlarge_22k']:
        backbone = build_convnext(modelname=args.backbone, pretrained=True, out_indices=tuple(return_interm_indices),backbone_dir=args.backbone_dir)
        bb_num_channels = backbone.dims[4 - len(return_interm_indices):]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))
    

    assert len(bb_num_channels) == len(return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"


    # 获取FPN类型
    fpn_type = getattr(args, 'fpn_type', 'standard')
    
    if fpn_type != 'standard':
        num_feature_levels=getattr(args, 'num_feature_levels', None)
        model = EnhancedJoiner(backbone, position_embedding, fpn_type, num_feature_levels)
        # 使用FPN时，输出通道数统一为256
        model.num_channels = [256] * len(return_interm_indices)
    else:
        model = Joiner(backbone, position_embedding)
        # 不使用FPN时，保持原始backbone的通道数
        model.num_channels = bb_num_channels
        
    assert isinstance(model.num_channels, List), "model.num_channels is expected to be a List but {}".format(type(model.num_channels))
    return model
