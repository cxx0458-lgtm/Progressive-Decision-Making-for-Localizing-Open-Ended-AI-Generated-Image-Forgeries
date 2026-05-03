import timm
import torch 
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import sys
import os
import time
import torchvision.utils as vutils
sys.path.append('.')
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor
import math
from functools import partial
from IMDLBenCo.registry import MODELS

class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self,conv_pretrain=False):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        
        return x , out
    def forward(self, image, mask=None, *args, **kwargs):

        feature,out = self.forward_features(image)

        return feature,out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class UpsampleConcatConvSegformer(nn.Module):
    def __init__(self):
        super(UpsampleConcatConvSegformer, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        # 上采样
        x1,x2,x3,x4 = inputs
        up2 = self.upsample1(x2)
        up3 = self.upsample2(x3)
        up4 = self.upsample3(x4)
        
        x = torch.cat([x1, up2, up3, up4], dim=1)
        return x




class MixVisionTransformer(nn.Module):
    def __init__(self,pretrain_path=None, img_size=512, patch_size=4, in_chans=3,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if pretrain_path is not None:
            print("Load segformer pretrain pth.")
            self.load_state_dict(torch.load(pretrain_path),
                                strict=False)
        original_first_layer = self.patch_embed1.proj
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
    
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.patch_embed1.proj = new_first_layer


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return x,outs

    def forward(self, x):
        x,outs = self.forward_features(x)
        return x,outs 


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleConcatConv(nn.Module):
    def __init__(self):
        super(UpsampleConcatConv, self).__init__()
        self.upsamplec2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)

        self.upsamples2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        # 上采样
        c1,c2,c3,c4,s1,s2,s3,s4 = inputs

        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)
        
        x = torch.cat([c1,c2,c3,c4,s1,s2,s3,s4 ], dim=1)
        features = [c1,c2,c3,c4,s1,s2,s3,s4]
        return x, features

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(192),
                                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(192, 768, kernel_size=1),
                                    nn.Conv2d(768, 192, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(192, 8,  kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x
class DiscriminativeChannelProjector(nn.Module):
    """
    1. 压缩/投影指导特征
    2. 额外预测一个 quality/guidance logits 用于辅助监督
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.quality_evaluator = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        proj_feat = self.channel_proj(x)
        quality_logits = self.quality_evaluator(proj_feat)
        quality_score = torch.sigmoid(quality_logits)
        return proj_feat, quality_logits, quality_score


def compute_balanced_quality_score_loss(quality_logits, gt_mask, lambda_aux=0.05):
    if gt_mask is None:
        return quality_logits.new_tensor(0.0)

    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    gt_mask = gt_mask.float()

    _, _, H_feat, W_feat = quality_logits.shape
    resized_gt = F.interpolate(gt_mask, size=(H_feat, W_feat), mode='nearest')

    num_positive = torch.sum(resized_gt == 1.0).float()
    num_negative = torch.sum(resized_gt == 0.0).float()
    pos_weight = num_negative / (num_positive + 1e-6)
    pos_weight = torch.clamp(pos_weight, max=10.0).reshape(1).to(quality_logits.device)

    balanced_bce = F.binary_cross_entropy_with_logits(
        quality_logits,
        resized_gt,
        pos_weight=pos_weight
    )
    return lambda_aux * balanced_bce

def soft_dice_loss(prob, target, eps=1e-6):
    """
    prob:   [B,1,H,W], already sigmoid probability
    target: [B,1,H,W] or [B,H,W]
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)

    prob = prob.float().reshape(prob.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)

    intersection = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

class MambaRefiner(nn.Module):
    def __init__(self, channels, steps=5, tau_low=0.05, tau_high=0.55, d_state=16, use_dep=True, use_uncertainty=True, use_boundary=True, use_region_constraint=True):
        super().__init__()
        self.steps = int(steps)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.use_boundary = use_boundary
        self.use_uncertainty = use_uncertainty
        self.use_region_constraint = use_region_constraint
        self.use_dep = use_dep
        if self.use_dep:
            self.feat_proj = DiscriminativeChannelProjector(channels, channels // 2)
            feat_dim = channels // 2
        else:
            self.feat_proj = None
            feat_dim = channels

        self.d_model = feat_dim + 3
        self.mamba_block = Mamba(
            d_model=self.d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )

        self.head = nn.Linear(self.d_model, 1)

        self.max_seq_len = 128 * 128
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.max_seq_len, self.d_model) * 0.02
        )

    def forward(self, p_raw, combined_feat):
        B, _, H, W = p_raw.shape
        seq_len = H * W
        if self.use_dep:
            proj_feat, quality_logits, quality_score = self.feat_proj(combined_feat)
        else:
            proj_feat = combined_feat
            quality_logits = torch.zeros(
                combined_feat.size(0), 1, combined_feat.size(2), combined_feat.size(3),
                device=combined_feat.device
            )
            quality_score = torch.ones_like(quality_logits)

        if self.use_boundary:
            grad_x = torch.abs(proj_feat[:, :, :, 1:] - proj_feat[:, :, :, :-1])
            grad_y = torch.abs(proj_feat[:, :, 1:, :] - proj_feat[:, :, :-1, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))

            edge_magnitude = torch.mean(grad_x + grad_y, dim=1, keepdim=True)
            edge_penalty = torch.exp(-edge_magnitude * 2.0)
        else:
            edge_penalty = torch.ones_like(p_raw)

        epsilon = 1e-6
        if self.use_uncertainty:
            uncertainty_map = -(p_raw * torch.log(p_raw + epsilon) +
                                (1 - p_raw) * torch.log(1 - p_raw + epsilon))
        else:
            uncertainty_map = torch.zeros_like(p_raw)

        mask_S = torch.sigmoid((p_raw - self.tau_high) * 10)
        mask_B = torch.sigmoid((self.tau_low - p_raw) * 10)
        mask_C = 1.0 - mask_S - mask_B

        p_current = p_raw.clone()
        p_outputs = []

        for _ in range(self.steps):
            state_input = torch.cat(
                [proj_feat, p_current, uncertainty_map, edge_penalty], dim=1
            )

            state_seq = state_input.flatten(2).transpose(1, 2)
            state_seq = state_seq + self.pos_embed[:, :seq_len, :]

            mamba_out = self.mamba_block(state_seq)

            p_diffused = torch.sigmoid(self.head(mamba_out))
            p_diffused = p_diffused.transpose(1, 2).view(B, 1, H, W)
            p_diffused = torch.nan_to_num(p_diffused, nan=0.5, posinf=1.0, neginf=0.0)
            p_diffused = torch.clamp(p_diffused, min=1e-6, max=1.0 - 1e-6)
           
            if self.use_region_constraint:
                p_current = mask_S * torch.max(p_current, p_diffused) + \
                            mask_C * p_diffused + \
                            mask_B * torch.min(p_current, p_diffused)
            else:
                p_current = p_diffused            

            p_current = torch.nan_to_num(p_current, nan=0.5, posinf=1.0, neginf=0.0)
            p_current = torch.clamp(p_current, min=1e-6, max=1.0 - 1e-6)

            p_outputs.append(p_current)

        mean_affinity = torch.tensor(0.0, device=p_raw.device)
        return (
            p_current,          # final refined prediction at 128x128
            mean_affinity,
            quality_logits,
            quality_score,
            p_outputs,          # progressive states after each step
            uncertainty_map,    # entropy map of p_raw
            edge_penalty,       # boundary-aware prior
            proj_feat,          # compact decision evidence
            p_raw               # initial state before refinement
        )

@MODELS.register_module()
class ProgressiveMesorch(nn.Module):
    def __init__(
        self,
        seg_pretrain_path=None,
        conv_pretrain=False,
        image_size=512,
        use_look_twice=True,
        lt_steps=2,
        lt_tau_low=0.10,
        lt_tau_high=0.60,
        lt_lambda_aux=0.01,
        lt_deep_dice_weight=1.0   # deep supervision 的 dice 权重
    ):
        super(ProgressiveMesorch, self).__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.upsample = UpsampleConcatConv()
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(4)] + [nn.Conv2d(64, 1, 1) for _ in range(4)])
        self.gate = ScoreNetwork()
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.use_look_twice = use_look_twice
        self.lt_lambda_aux = lt_lambda_aux

        self.lt_deep_dice_weight = float(lt_deep_dice_weight)

        if self.use_look_twice:
            self.fusion_weight = nn.Parameter(torch.tensor(0.0))
            self.look_twice = MambaRefiner(
                channels=8,
                steps=lt_steps,
                tau_low=lt_tau_low,
                tau_high=lt_tau_high,
                d_state=16
            )

    def forward(self, image, mask=None, *args, **kwargs):
        high_freq = self.high_dct(image)
        low_freq = self.low_dct(image)
        input_high = torch.concat([image,high_freq],dim=1)
        input_low = torch.concat([image,low_freq],dim=1)
        input_all = torch.concat([image,high_freq,low_freq],dim=1)
        _,outs1 = self.convnext(input_high)
        _,outs2 = self.segformer(input_low)

        inputs = outs1 + outs2
        x, features = self.upsample(inputs)
        # print(x.shape)
        gate_outputs = self.gate(input_all)

        reduced = torch.cat([self.inverse[i](features[i]) for i in range(8)], dim=1)
        # print(reduced.shape)
        pred_logits_128 = torch.sum(gate_outputs * reduced, dim=1, keepdim=True) 
        
        # 提取基础预测 (Baseline)
        pred_logits_baseline = self.resize(pred_logits_128)
        mask_pred_baseline = torch.sigmoid(pred_logits_baseline)
        
        # +++ Look Twice 分支逻辑 +++
        if self.use_look_twice:
            p_raw_128 = torch.sigmoid(pred_logits_128)
            # ===== region-aware masks for visualization =====
            tau_low = float(self.look_twice.tau_low)
            tau_high = float(self.look_twice.tau_high)

            # soft masks
            mask_S_soft_128 = torch.sigmoid((p_raw_128 - tau_high) * 10.0)
            mask_B_soft_128 = torch.sigmoid((tau_low - p_raw_128) * 10.0)
            mask_C_soft_128 = torch.clamp(1.0 - mask_S_soft_128 - mask_B_soft_128, min=0.0, max=1.0)

            # hard masks
            mask_S_hard_128 = (p_raw_128 > tau_high).float()
            mask_B_hard_128 = (p_raw_128 < tau_low).float()
            mask_C_hard_128 = 1.0 - mask_S_hard_128 - mask_B_hard_128

            (
                mask_pred_128_lt,
                mean_affinity,
                quality_logits,
                quality_score,
                p_outputs,
                uncertainty_map,
                edge_penalty,
                proj_feat,
                p_raw_state
            ) = self.look_twice(p_raw_128, reduced)
            mask_pred_lt = self.resize(mask_pred_128_lt)

            alpha = torch.sigmoid(self.fusion_weight)
            mask_pred = alpha * mask_pred_lt + (1.0 - alpha) * mask_pred_baseline
            mask_pred = torch.clamp(mask_pred, min=1e-6, max=1.0 - 1e-6)

            if mask is not None:
                # ===== 1) final fused prediction loss =====
                bce_loss = F.binary_cross_entropy(mask_pred, mask.float())
                dice_loss = soft_dice_loss(mask_pred, mask.float())
                final_loss = bce_loss 

                # ===== 2) auxiliary quality/guidance supervision =====
                quality_loss = compute_balanced_quality_score_loss(
                    quality_logits, mask, lambda_aux=self.lt_lambda_aux
                )

                # ===== 3) deep supervision on iterative refinement =====
                loss_deep = 0.0
                for i, p_step in enumerate(p_outputs):
                    p_step_up = F.interpolate(
                        p_step, size=mask.shape[-2:], mode='bilinear', align_corners=False
                    )
                    p_step_up = torch.nan_to_num(p_step_up, nan=0.5, posinf=1.0, neginf=0.0)
                    p_step_up = torch.clamp(p_step_up, min=1e-6, max=1.0 - 1e-6)

                    weight = (i + 1) / len(p_outputs)
                    step_bce = F.binary_cross_entropy(p_step_up, mask.float())
                    step_dice = soft_dice_loss(p_step_up, mask.float())
                    loss_deep += weight * weight * (step_bce +  step_dice)

                loss_deep = self.lt_deep_dice_weight * loss_deep

                # ===== total =====
                loss = final_loss + quality_loss + loss_deep
            else:
                loss = torch.tensor(0.0, device=mask_pred.device)
                bce_loss = torch.tensor(0.0, device=mask_pred.device)
                dice_loss = torch.tensor(0.0, device=mask_pred.device)
                mean_affinity = torch.tensor(0.0, device=mask_pred.device)
                quality_loss = torch.tensor(0.0, device=mask_pred.device)
                loss_deep = torch.tensor(0.0, device=mask_pred.device)
                final_loss = torch.tensor(0.0, device=mask_pred.device)

        else:
            mask_pred = mask_pred_baseline
            alpha = torch.tensor(0.0, device=mask_pred.device)
            mean_affinity = torch.tensor(0.0, device=mask_pred.device)
            quality_loss = torch.tensor(0.0, device=mask_pred.device)
            loss_deep = torch.tensor(0.0, device=mask_pred.device)

            if mask is not None:
                loss = self.loss_fn(pred_logits_baseline, mask)
                bce_loss = loss
                dice_loss = torch.tensor(0.0, device=mask_pred.device)
                final_loss = loss
            else:
                loss = torch.tensor(0.0, device=pred_logits_baseline.device)
                bce_loss = torch.tensor(0.0, device=pred_logits_baseline.device)
                dice_loss = torch.tensor(0.0, device=pred_logits_baseline.device)
                final_loss = torch.tensor(0.0, device=pred_logits_baseline.device)

        # +++  TensorBoard 可视化接口 +++
        output_dict = {
            "backward_loss": loss,
            "pred_mask": mask_pred,
            "pred_label": None,
            
            # 1. 监控标量 (Scalars)：会自动画出折线图
            "visual_loss": {
                "predict_loss": loss,
                "bce_loss": bce_loss if mask is not None else torch.tensor(0.0).to(mask_pred.device),
                "dice_loss": dice_loss if mask is not None else torch.tensor(0.0).to(mask_pred.device),
                "final_mask_loss": final_loss if mask is not None else torch.tensor(0.0).to(mask_pred.device),
                "lt_deep_dice_weight": torch.tensor(self.lt_deep_dice_weight).to(mask_pred.device),
            },
            
            # 2. 监控图像 (Images)：会自动在 TensorBoard 显示图片
            "visual_image": {
                "01_ground_truth": mask if mask is not None else mask_pred,
                "02_baseline_pred": mask_pred_baseline, # 看看原本预测成啥样
                "03_final_fused_pred": mask_pred        # 看看最终融合成了啥样
            }
        }
        
        
        if self.use_look_twice:
            output_dict["visual_loss"]["lt_fusion_alpha"] = alpha
            output_dict["visual_loss"]["lt_mean_affinity"] = mean_affinity
            output_dict["visual_loss"]["lt_quality_loss"] = quality_loss
            output_dict["visual_loss"]["lt_deep_loss"] = loss_deep if mask is not None else torch.tensor(0.0).to(mask_pred.device)
            # ===== 1. quality =====
            output_dict["visual_image"]["04_quality_score"] = self.resize(quality_score)

            # ===== 2. uncertainty =====
            output_dict["visual_image"]["05_uncertainty_map"] = self.resize(uncertainty_map)

            # ===== 3. boundary =====
            output_dict["visual_image"]["06_boundary_prior"] = self.resize(edge_penalty)

            # ===== 4. decision evidence（通道平均）=====
            proj_feat_vis = torch.mean(proj_feat, dim=1, keepdim=True)
            output_dict["visual_image"]["07_decision_evidence"] = self.resize(proj_feat_vis)

            # ===== 5. P0 =====
            output_dict["visual_image"]["08_p0_initial_state"] = self.resize(p_raw_state)
            output_dict["visual_image"]["09_refined_only_pred"] = mask_pred_lt

            # ===== 6. region-aware masks =====
            output_dict["visual_image"]["10_region_manip_soft"] = self.resize(mask_S_soft_128)
            output_dict["visual_image"]["11_region_uncertain_soft"] = self.resize(mask_C_soft_128)
            output_dict["visual_image"]["12_region_background_soft"] = self.resize(mask_B_soft_128)

            output_dict["visual_image"]["13_region_manip_hard"] = self.resize(mask_S_hard_128)
            output_dict["visual_image"]["14_region_uncertain_hard"] = self.resize(mask_C_hard_128)
            output_dict["visual_image"]["15_region_background_hard"] = self.resize(mask_B_hard_128)

            # ===== 7. progressive steps =====
            for i, p_step in enumerate(p_outputs):
                step_up = F.interpolate(
                    p_step,
                    size=mask_pred.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                output_dict["visual_image"][f"step_{i+1:02d}_pred"] = step_up
        return output_dict

if __name__ == "__main__":
    print(MODELS)