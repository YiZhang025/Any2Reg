"""
Any2RegNet and dependencies. Self-contained for inference; uses model_utils for warp/CTE.
"""
from typing import List, Literal, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

from .model_utils import (
    compose_flow,
    compute_cte_template,
    compute_cte_template_batched,
    warp_2d,
)


# ---------- CTE / Mean convs and set encoders ----------
class CTEConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.PReLU()

    def compute_cte_weights(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_flat = x.reshape(B, N, -1)
        x_norm = x_flat - x_flat.mean(dim=2, keepdim=True)
        x_norm = x_norm / (x_norm.norm(dim=2, keepdim=True) + 1e-8)
        corr = torch.bmm(x_norm, x_norm.transpose(1, 2))
        _eigvals, eigvecs = torch.linalg.eigh(corr)
        weights = eigvecs[:, :, -1]
        weights = torch.abs(weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        weights = self.compute_cte_weights(x)
        weights_expanded = weights.view(B, N, 1, 1, 1)
        group_feat = (x * weights_expanded).sum(dim=1).unsqueeze(1).expand(-1, N, -1, -1, -1)
        x_cat = torch.cat([x, group_feat], dim=2).view(B * N, 2 * C, H, W)
        out = self.act(self.norm(self.conv(x_cat)))
        return out.view(B, N, -1, H, W)


class MeanConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        group_feat = x.mean(dim=1, keepdim=True).expand(-1, N, -1, -1, -1)
        x_cat = torch.cat([x, group_feat], dim=2).view(B * N, 2 * C, H, W)
        out = self.act(self.norm(self.conv(x_cat)))
        return out.view(B, N, -1, H, W)


def _down_view(x, down):
    return down(x.view(x.shape[0] * x.shape[1], *x.shape[2:])).view(
        x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, x.shape[4] // 2
    )


class SetEncoderMean2D(nn.Module):
    def __init__(self, in_channels: int = 1, channel_num: int = 16, use_checkpoint: bool = False):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2)
        self.block1 = MeanConv2d(in_channels, channel_num)
        self.block2 = MeanConv2d(channel_num, channel_num * 2)
        self.block3 = MeanConv2d(channel_num * 2, channel_num * 4)
        self.block4 = MeanConv2d(channel_num * 4, channel_num * 8)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.block1(x)
        x = _down_view(x1, self.down)
        x2 = self.block2(x)
        x = _down_view(x2, self.down)
        x3 = self.block3(x)
        x = _down_view(x3, self.down)
        x4 = self.block4(x)
        return [x1, x2, x3, x4]


class SetEncoderCTE2D(nn.Module):
    def __init__(self, in_channels: int = 1, channel_num: int = 16, use_checkpoint: bool = False):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2)
        self.block1 = CTEConv2d(in_channels, channel_num)
        self.block2 = CTEConv2d(channel_num, channel_num * 2)
        self.block3 = CTEConv2d(channel_num * 2, channel_num * 4)
        self.block4 = CTEConv2d(channel_num * 4, channel_num * 8)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.block1(x)
        x = _down_view(x1, self.down)
        x2 = self.block2(x)
        x = _down_view(x2, self.down)
        x3 = self.block3(x)
        x = _down_view(x3, self.down)
        x4 = self.block4(x)
        return [x1, x2, x3, x4]


class FeatureFusionModule(nn.Module):
    def __init__(self, channels: int, fusion_mode: str = "weighted_add", use_checkpoint: bool = False):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == "weighted_add":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_mode == "attention":
            self.attention = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, 1),
                nn.Softmax(dim=1),
            )
        elif fusion_mode == "concat_conv":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_mode != "add":
            raise ValueError(f"fusion_mode must be add, weighted_add, attention, concat_conv; got {fusion_mode}")

    def forward(self, img_feat: torch.Tensor, feat_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if feat_feat is None:
            return img_feat
        if self.fusion_mode == "add":
            return img_feat + feat_feat
        if self.fusion_mode == "weighted_add":
            alpha = torch.sigmoid(self.alpha)
            return alpha * img_feat + (1 - alpha) * feat_feat
        if self.fusion_mode == "attention":
            concat = torch.cat([img_feat, feat_feat], dim=1)
            attn = self.attention(concat)
            return attn[:, 0:1] * img_feat + attn[:, 1:2] * feat_feat
        concat = torch.cat([img_feat, feat_feat], dim=1)
        return self.fusion_conv(concat)


class LocalCorrelation2D(nn.Module):
    def __init__(self, max_disp: int = 1, normalize: bool = True):
        super().__init__()
        self.max_disp = max_disp
        self.normalize = normalize

    def forward(self, F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F1.shape
        d = self.max_disp
        if self.normalize:
            F1 = F1 / (F1.norm(dim=1, keepdim=True) + 1e-6)
            F2 = F2 / (F2.norm(dim=1, keepdim=True) + 1e-6)
        pad = (d, d, d, d)
        F2_pad = F.pad(F2, pad, mode="replicate")
        corrs = []
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                y0, x0 = dy + d, dx + d
                y1, x1 = y0 + H, x0 + W
                F2_shift = F2_pad[:, :, y0:y1, x0:x1]
                corrs.append((F1 * F2_shift).sum(dim=1, keepdim=True))
        return torch.cat(corrs, dim=1)


# ---------- Decoder building blocks ----------
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def _forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)))

    def forward(self, x):
        return checkpoint.checkpoint(self._forward, x) if self.use_checkpoint and x.requires_grad else self._forward(x)


class RegHead2D(nn.Module):
    def __init__(self, in_channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.reg_head = nn.Conv2d(in_channels, 2, 3, 1, 1)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros_like(self.reg_head.bias))

    def forward(self, x):
        return checkpoint.checkpoint(self.reg_head, x) if self.use_checkpoint and x.requires_grad else self.reg_head(x)


class PatchExpanding2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.up_conv(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class ResizeFlow2D(nn.Module):
    def __init__(self, resize_factor: float, mode: str = "bilinear"):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor < 1:
            x = F.interpolate(x, scale_factor=self.factor, mode=self.mode, align_corners=True)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, scale_factor=self.factor, mode=self.mode, align_corners=True)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class CALayer2D(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4, use_bias: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels // reduction, 1, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels // reduction, num_channels, 1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: (B,H,W,C)
        x = x_in.permute(0, 3, 1, 2).mean(dim=(2, 3), keepdim=True)
        w = self.sigmoid(self.conv2(self.relu(self.conv1(x))))  # (B,C,1,1)
        return x_in * w.permute(0, 2, 3, 1)


class RCAB2D(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4, lrelu_slope: float = 0.2, use_bias: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=use_bias)
        self.act = nn.LeakyReLU(lrelu_slope, inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=use_bias)
        self.ca = CALayer2D(num_channels, reduction, use_bias)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ca(x)
        return x + shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint.checkpoint(self._forward, x) if self.use_checkpoint and x.requires_grad else self._forward(x)


class SpatialGatingUnit2D(nn.Module):
    def __init__(self, c: int, n: int, use_bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(n, n, bias=use_bias)
        self.norm = nn.LayerNorm(c // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        v = self.norm(v)
        v = self.dense(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return u * (v + 1.0)


class WinGmlpLayer2D(nn.Module):
    def __init__(self, win_size: Tuple[int, int], num_channels: int, factor: int = 2, use_bias: bool = True):
        super().__init__()
        self.fh, self.fw = win_size
        self.norm = nn.LayerNorm(num_channels)
        self.in_proj = nn.Linear(num_channels, num_channels * factor, bias=use_bias)
        self.gelu = nn.GELU()
        self.sgu = SpatialGatingUnit2D(num_channels * factor, self.fh * self.fw, use_bias)
        self.out_proj = nn.Linear(num_channels * factor // 2, num_channels, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        pad_b = (self.fh - H % self.fh) % self.fh
        pad_r = (self.fw - W % self.fw) % self.fw
        x_chw = x.permute(0, 3, 1, 2)
        if pad_b or pad_r:
            x_chw = F.pad(x_chw, (0, pad_r, 0, pad_b))
        _, _, H_pad, W_pad = x_chw.shape
        gh, gw = H_pad // self.fh, W_pad // self.fw
        x_win = einops.rearrange(x_chw, "b c (gh fh) (gw fw) -> b (gh gw) (fh fw) c", gh=gh, gw=gw, fh=self.fh, fw=self.fw)
        shortcut = x_win
        x_win = self.out_proj(self.sgu(self.gelu(self.in_proj(self.norm(x_win))))) + shortcut
        x_chw = einops.rearrange(x_win, "b (gh gw) (fh fw) c -> b c (gh fh) (gw fw)", gh=gh, gw=gw, fh=self.fh, fw=self.fw)
        x_chw = x_chw[:, :, :H, :W].contiguous()
        return x_chw.permute(0, 2, 3, 1)


class MultiWinMlpLayer2D(nn.Module):
    def __init__(self, num_channels: int, use_bias: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(num_channels)
        self.win1 = WinGmlpLayer2D((3, 3), num_channels, use_bias=use_bias)
        self.win2 = WinGmlpLayer2D((5, 5), num_channels, use_bias=use_bias)
        self.win3 = WinGmlpLayer2D((7, 7), num_channels, use_bias=use_bias)
        self.reweight = MLP(num_channels, num_channels // 4, num_channels * 3)
        self.out_proj = nn.Linear(num_channels, num_channels, bias=use_bias)

    def _forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.norm(x_in)
        x1, x2, x3 = self.win1(x), self.win2(x), self.win3(x)
        a = (x1 + x2 + x3).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(x.shape[0], -1, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = x1 * a[0] + x2 * a[1] + x3 * a[2]
        return self.out_proj(x) + x_in

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return checkpoint.checkpoint(self._forward, x_in) if self.use_checkpoint and x_in.requires_grad else self._forward(x_in)


class CMWMLPBlock2D(nn.Module):
    def __init__(self, in_channels: int, num_channels: int, use_corr: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_corr = use_corr
        self.use_checkpoint = use_checkpoint
        if use_corr:
            self.corr = LocalCorrelation2D(max_disp=1)
            corr_ch = 9
            self.conv = nn.Conv2d(in_channels * 2 + corr_ch, num_channels, 3, 1, 1)
        else:
            self.corr = None
            self.conv = nn.Conv2d(in_channels * 2, num_channels, 3, 1, 1)
        self.mlp_layer = MultiWinMlpLayer2D(num_channels, use_checkpoint=use_checkpoint)
        self.channel_att = RCAB2D(num_channels, use_checkpoint=use_checkpoint)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.use_corr:
            x = torch.cat([x1, self.corr(x1, x2), x2], dim=1)
        else:
            x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.mlp_layer(x)
        x = self.channel_att(x)
        return x.permute(0, 3, 1, 2) + shortcut


class MLPDecoder2D(nn.Module):
    def __init__(self, in_channels: int, channel_num: int, use_checkpoint: bool = False):
        super().__init__()
        self.mlp_11 = CMWMLPBlock2D(in_channels, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_12 = CMWMLPBlock2D(in_channels * 2, channel_num * 2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_13 = CMWMLPBlock2D(in_channels * 4, channel_num * 4, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_14 = CMWMLPBlock2D(in_channels * 8, channel_num * 8, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_21 = CMWMLPBlock2D(channel_num, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_22 = CMWMLPBlock2D(channel_num * 2, channel_num * 2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_23 = CMWMLPBlock2D(channel_num * 4, channel_num * 4, use_corr=True, use_checkpoint=use_checkpoint)
        self.upsample_1 = PatchExpanding2D(channel_num * 2)
        self.upsample_2 = PatchExpanding2D(channel_num * 4)
        self.upsample_3 = PatchExpanding2D(channel_num * 8)
        self.resize_flow = ResizeFlow2D(2, "bilinear")
        self.reghead_1 = RegHead2D(channel_num)
        self.reghead_2 = RegHead2D(channel_num * 2)
        self.reghead_3 = RegHead2D(channel_num * 4)
        self.reghead_4 = RegHead2D(channel_num * 8)

    def forward(self, x_fix_list: List[torch.Tensor], x_mov_list: List[torch.Tensor]) -> torch.Tensor:
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix_list
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov_list
        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4)
        flow_4_up = self.resize_flow(flow_4)
        x_mov_3_warp = warp_2d(x_mov_3, flow_4_up)
        x_3 = self.mlp_23(self.mlp_13(x_fix_3, x_mov_3_warp), self.upsample_3(x_4))
        flow_3 = compose_flow(flow_4_up, self.reghead_3(x_3))
        flow_3_up = self.resize_flow(flow_3)
        x_mov_2_warp = warp_2d(x_mov_2, flow_3_up)
        x_2 = self.mlp_22(self.mlp_12(x_fix_2, x_mov_2_warp), self.upsample_2(x_3))
        flow_2 = compose_flow(flow_3_up, self.reghead_2(x_2))
        flow_2_up = self.resize_flow(flow_2)
        x_mov_1_warp = warp_2d(x_mov_1, flow_2_up)
        x_1 = self.mlp_21(self.mlp_11(x_fix_1, x_mov_1_warp), self.upsample_1(x_2))
        flow_1 = compose_flow(flow_2_up, self.reghead_1(x_1))
        return flow_1


# ---------- Any2RegNet ----------
class Any2RegNet(nn.Module):
    """Siamese Set Encoder + CorrMLP Decoder. Input: (N,C,H,W), optional feature_maps (N,Cf,H,W)."""

    def __init__(
        self,
        in_channels: int = 1,
        enc_channels: int = 16,
        dec_channels: int = 16,
        use_checkpoint: bool = True,
        feat_in_channels: Optional[int] = None,
        fusion_mode: str = "weighted_add",
        num_iters: int = 1,
        encoder_aggregation: Literal["cte", "mean"] = "cte",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_in_channels = feat_in_channels
        self.fusion_mode = fusion_mode
        self.num_iters = num_iters
        self.encoder_aggregation = encoder_aggregation
        self.cte_downsample = 1

        SetEncoderClass = SetEncoderMean2D if encoder_aggregation == "mean" else SetEncoderCTE2D
        self.img_set_encoder = SetEncoderClass(in_channels=in_channels, channel_num=enc_channels, use_checkpoint=use_checkpoint)
        if feat_in_channels is not None:
            self.feat_set_encoder = SetEncoderClass(in_channels=feat_in_channels, channel_num=enc_channels, use_checkpoint=use_checkpoint)
            enc_ch_list = [enc_channels, enc_channels * 2, enc_channels * 4, enc_channels * 8]
            self.fusion_modules = nn.ModuleList([FeatureFusionModule(ch, fusion_mode=fusion_mode, use_checkpoint=use_checkpoint) for ch in enc_ch_list])
        else:
            self.feat_set_encoder = None
            self.fusion_modules = None
        self.decoder = MLPDecoder2D(enc_channels, dec_channels, use_checkpoint=use_checkpoint)

    def _encode_siamese(self, images: torch.Tensor, feature_maps: Optional[torch.Tensor]) -> List[torch.Tensor]:
        N = images.shape[0]
        x_img = images.unsqueeze(0)
        img_feats = self.img_set_encoder(x_img)
        if feature_maps is not None and self.feat_set_encoder is not None:
            x_feat = feature_maps.unsqueeze(0)
            feat_feats = self.feat_set_encoder(x_feat)
            fused_feats = []
            for img_f, feat_f, fusion in zip(img_feats, feat_feats, self.fusion_modules):
                B, N_seq, C, H, W = img_f.shape
                img_f_flat = img_f.view(B * N_seq, C, H, W)
                feat_f_flat = feat_f.view(B * N_seq, C, H, W)
                fused_flat = fusion(img_f_flat, feat_f_flat)
                fused_feats.append(fused_flat.view(B, N_seq, C, H, W))
            return fused_feats
        return img_feats

    def _single_forward(
        self,
        images: torch.Tensor,
        feats_list: List[torch.Tensor],
        prev_flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = images.shape[0]
        if prev_flow is not None:
            warped_feats_list = []
            for feats in feats_list:
                _, _, c, h, w = feats.shape
                scale_h, scale_w = h / images.shape[2], w / images.shape[3]
                scaled_flow = F.interpolate(prev_flow, size=(h, w), mode="bilinear", align_corners=True)
                scaled_flow = scaled_flow * torch.tensor([scale_w, scale_h], device=scaled_flow.device).view(1, 2, 1, 1)
                warped_feats_list.append(warp_2d(feats[0], scaled_flow).unsqueeze(0))
            feats_list = warped_feats_list

        t_list = [compute_cte_template_batched(f) for f in feats_list]
        fix_list = [t_list[k].expand(N, -1, -1, -1) for k in range(4)]
        mov_list = [feats_list[k][0] for k in range(4)]
        flow_or_res = self.decoder(fix_list, mov_list)
        flow = compose_flow(prev_flow, flow_or_res) if prev_flow is not None else flow_or_res
        warped = warp_2d(images, flow)
        template = compute_cte_template(warped, downsample=self.cte_downsample)
        return warped, flow, template

    def forward(
        self,
        images: torch.Tensor,
        feature_maps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.dim() == 5:
            images = images.squeeze(0)
        N, C, H, W = images.shape
        assert C == self.in_channels
        feats_list = self._encode_siamese(images, feature_maps)
        flow = None
        for k in range(self.num_iters):
            if k == 0:
                warped, flow, template = self._single_forward(images, feats_list, prev_flow=None)
            else:
                feature_maps_iter = warp_2d(feature_maps, flow) if feature_maps is not None else None
                feats_list_iter = self._encode_siamese(warped, feature_maps_iter)
                warped, flow, template = self._single_forward(images, feats_list_iter, prev_flow=flow)
        return warped, flow, template


def create_any2regnet(
    sample_images: torch.Tensor,
    in_channels: int = 1,
    enc_channels: int = 16,
    dec_channels: int = 16,
    num_iters: int = 1,
    feat_in_channels: Optional[int] = None,
    fusion_mode: str = "weighted_add",
    encoder_aggregation: str = "cte",
    use_checkpoint: bool = False,
) -> Any2RegNet:
    """Build Any2RegNet for inference (same interface as train_unified create_model_from_baseline for any2regnet)."""
    if feat_in_channels is None and sample_images.dim() == 4:
        feat_in_channels = 1
    model = Any2RegNet(
        in_channels=in_channels,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        use_checkpoint=use_checkpoint,
        feat_in_channels=feat_in_channels,
        fusion_mode=fusion_mode,
        num_iters=num_iters,
        encoder_aggregation=encoder_aggregation,
    )
    return model
