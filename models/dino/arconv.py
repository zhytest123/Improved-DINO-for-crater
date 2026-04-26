import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Optional
from util.misc import NestedTensor, is_main_process
from .backbone import FrozenBatchNorm2d

########################################
# 原始 ARConv 与 ARNet 代码可放这里（你提供的实现）。
# 这里只保留核心 ARConv（去掉与 ARNet 结构无关的上采样/下采样）。
########################################

class ARConv(nn.Module):
    """
    你提供的 ARConv 实现。
    仅做极少量安全性调整：移除 hook 注册中已废弃的 register_full_backward_hook 警告可选。
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 l_max=9, w_max=9, flag=False, modulation=True):
        super().__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]  # 组合 (3,3)(3,5)...(7,7)
        #self.i_list = [33, 35, 53]
        
        self.convs = nn.ModuleList([
            nn.Conv2d(inc, outc,
                      kernel_size=(i // 10, i % 10),
                      stride=(i // 10, i % 10),
                      padding=0)
            for i in self.i_list
        ])

        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, 1, 1),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, 1, 1)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 固定最终自适应窗口 (N_X, N_Y) 缓存
        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32),
                                         requires_grad=False)

    def forward(self, x, epoch:int, hw_range:List[int]):
        # hw_range: [最小尺寸估计, 最大尺寸估计] 或自定义策略[1,18]
        assert isinstance(hw_range, list) and len(hw_range) == 2
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1

        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)

        l = self.l_conv(offset) * (hw_range[1] - 1) + 1
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1

        if epoch <= 100:
            mean_l = l.mean()
            mean_w = w.mean()
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)

            def phi(v):
                if v % 2 == 0:
                    v -= 1
                return v
            N_X = phi(max(3, min(7, N_X)))
            N_Y = phi(max(3, min(7, N_Y)))
            if epoch == 100:
                self.reserved_NXY = nn.Parameter(torch.tensor([N_X, N_Y],
                                             dtype=torch.int32,
                                             device=x.device),
                                                 requires_grad=False)
        else:
            N_X = int(self.reserved_NXY[0])
            N_Y = int(self.reserved_NXY[1])

        N = N_X * N_Y
        l = l.repeat(1, N, 1, 1)
        w = w.repeat(1, N, 1, 1)
        offset_cat = torch.cat([l, w], 1).to(x.device)

        if self.padding:
            x = self.zero_padding(x)
        
        p = self._get_p(offset_cat, x.dtype, N_X, N_Y)  # (b,2N,h,w)
        p = p.permute(0, 2, 3, 1)
        q_lt = p.floor()
        q_rb = q_lt + 1

        def clamp_coord(q):
            return torch.cat([
                torch.clamp(q[..., :N], 0, x.size(2) - 1),
                torch.clamp(q[..., N:], 0, x.size(3) - 1)
            ], -1).long()

        q_lt = clamp_coord(q_lt)
        q_rb = clamp_coord(q_rb)
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], -1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (g_lt.unsqueeze(1) * x_q_lt +
                    g_rb.unsqueeze(1) * x_q_rb +
                    g_lb.unsqueeze(1) * x_q_lb +
                    g_rt.unsqueeze(1) * x_q_rt)

        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        conv_idx = self.i_list.index(N_X * 10 + N_Y)
        x_offset = self.convs[conv_idx](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y, device):
        # 兼容旧版 torch，无 indexing 参数，默认行为等同 'ij'
        px = torch.arange(-(n_x - 1)//2, (n_x - 1)//2 + 1, device=device, dtype=dtype)
        py = torch.arange(-(n_y - 1)//2, (n_y - 1)//2 + 1, device=device, dtype=dtype)
        px, py = torch.meshgrid(px, py)  # 旧版无 indexing
        p_n = torch.cat([px.reshape(-1), py.reshape(-1)], 0).view(1, 2 * N, 1, 1)
        return p_n

    def _get_p_0(self, h, w, N, dtype, device):
        px = torch.arange(1, h * self.stride + 1, self.stride, device=device, dtype=dtype)
        py = torch.arange(1, w * self.stride + 1, self.stride, device=device, dtype=dtype)
        px, py = torch.meshgrid(px, py)
        px = px.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        py = py.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([px, py], 1)

    def _get_p(self, offset, dtype, n_x, n_y):
        device = offset.device
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], 1)
        L = L / n_x
        W = W / n_y
        offset_norm = torch.cat([L, W], 1)
        p_n = self._get_p_n(N, dtype, n_x, n_y, device).repeat(1, 1, h, w)
        p_0 = self._get_p_0(h, w, N, dtype, device)
        return p_0 + offset_norm * p_n

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        pw = x.size(3)
        c = x.size(1)
        x_flat = x.view(b, c, -1)
        idx = q[..., :N] * pw + q[..., N:]
        idx = idx.unsqueeze(1).expand(-1, c, -1, -1, -1).reshape(b, c, -1)
        x_offset = x_flat.gather(-1, idx).view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_off, n_x, n_y):
        b, c, h, w, N = x_off.size()
        rows = []
        for s in range(0, N, n_y):
            rows.append(x_off[..., s:s + n_y].reshape(b, c, h, w * n_y))
        x_cat = torch.cat(rows, -1)
        return x_cat.view(b, c, h * n_x, w * n_y)


########################################
# 适配包装：保持与 nn.Conv2d 一致的 forward(x)
########################################
class ARConvAdapter(nn.Module):
    """
    将需要 (x, epoch, hw_range) 的 ARConv 包装成只接收 x。
    在外部通过 set_runtime(epoch, hw_range) 设置运行期参数。
    """
    def __init__(self, orig_conv: nn.Conv2d):
        super().__init__()
        self.arconv = ARConv(
            inc=orig_conv.in_channels,
            outc=orig_conv.out_channels,
            kernel_size=3,
            padding=1,
            stride=orig_conv.stride[0] if isinstance(orig_conv.stride, tuple) else orig_conv.stride
        )
        # 运行期状态
        self._epoch: int = 0
        self._hw_range: List[int] = [1, 3]

    def set_runtime(self, epoch:int, hw_range:List[int]):
        self._epoch = int(epoch)
        self._hw_range = hw_range

    def forward(self, x):
        return self.arconv(x, self._epoch, self._hw_range)




def replace_conv3x3_with_arconv_policy(model: nn.Module,
                                       mode: str = 'layer4_only',
                                       verbose: bool = False,
                                       only_stride1: bool = False):
    """
    支持替换模式:
      layer4_only  -> 只替换 layer4.*.conv2
      layer1_2_3   -> 替换 layer1/layer2/layer3.*.conv2
      all_layers   -> 替换 layer1-4 所有 Bottleneck 的 conv2
      layer1_only  -> 只替换 layer1.*.conv2
      layer1_2_only -> 替换 layer1/layer2.*.conv2
    规则:
      1. 只替换 Bottleneck 中名字为 conv2 的 3x3 普通卷积
      2. 仅要求 kernel_size==(3,3), groups==1, dilation==(1,1)
      3. only_stride1=True 时再加约束 stride==(1,1)
    """
    if mode == 'layer4_only':
        target_stages = {'layer4'}
    elif mode == 'layer1_2_3':
        target_stages = {'layer1', 'layer2', 'layer3'}
    elif mode == 'all_layers':
        target_stages = {'layer1', 'layer2', 'layer3', 'layer4'}
    elif mode == 'layer1_only':
        target_stages = {'layer1'}
    elif mode == 'layer1_2_only':
        target_stages = {'layer1', 'layer2'}
    else:
        raise ValueError(f"不支持的 mode: {mode}. 允许值: layer4_only | layer1_2_3 | all_layers | layer1_only | layer1_2_only")

    replaced = 0
    
    for full_name, module in model.named_modules():
        parts = full_name.split('.')
        if len(parts) != 3:
            continue
        stage, block, lname = parts
        if stage not in target_stages or lname != 'conv2':
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if not (module.kernel_size == (3,3) and module.groups == 1 and module.dilation == (1,1)):
            continue
        if only_stride1 and module.stride != (1,1):
            continue
        parent_name = '.'.join(parts[:-1])
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, 'conv2', ARConvAdapter(module))
        replaced += 1
        if verbose:
            print(f"[ARConv] replaced: {full_name}")
    if verbose:
        print(f"[ARConv] mode={mode}, only_stride1={only_stride1}, total replaced={replaced}")


class ARConvBackbone(nn.Module):
    """
    与 SAConvBackbone 结构一致：构建 ResNet 并替换 3x3 卷积为 ARConvAdapter
    """
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices: List[int],
                 batch_norm_type='FrozenBatchNorm2d',
                 arconv_mode: str = 'layer4_only',
                 arconv_verbose: bool = False,
                 arconv_only_stride1: bool = False):
        super().__init__()
        bn = FrozenBatchNorm2d if batch_norm_type == 'FrozenBatchNorm2d' else nn.BatchNorm2d
        assert name in ['resnet50', 'resnet101']
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=bn
        )
        replace_conv3x3_with_arconv_policy(
            backbone,
            mode=arconv_mode,
            verbose=arconv_verbose,
            only_stride1=arconv_only_stride1
        )

        # 冻结策略
        for pname, p in backbone.named_parameters():
            if (not train_backbone) or all(s not in pname for s in ['layer2','layer3','layer4']):
                p.requires_grad_(False)
        assert return_interm_indices in [[0,1,2,3],[1,2,3],[3]]
        all_channels = [256,512,1024,2048]
        self.num_channels = all_channels[4 - len(return_interm_indices):]
        return_layers = {}
        for idx, lid in enumerate(return_interm_indices):
            return_layers.update({f"layer{5 - len(return_interm_indices) + idx}": f"{lid}"})
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def set_runtime(self, epoch:int, hw_range:List[int]):
        """
        训练循环中每个 iteration / epoch 调用。
        向所有 ARConvAdapter 注入运行期参数。
        """
        def apply_runtime(m: nn.Module):
            if isinstance(m, ARConvAdapter):
                m.set_runtime(epoch, hw_range)
        self.apply(apply_runtime)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None
        for name, x in xs.items():
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out