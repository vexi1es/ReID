"""
Модель re-id: DINOv2 (ViT) или DINOv3 ConvNeXt (timm) + BNNeck + ArcFace
(рецепт reid-strong-baseline / AI City).

  признак = concat(CLS, mean(patch))  ->  BN  ->  L2   (для поиска)
  обучение: ArcFace по BN-признаку + batch-hard triplet по признаку до BN
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class ReIDModel(nn.Module):
    """
    backbone="facebook/dinov2-base" — ViT из transformers, признак concat(CLS, mean(patch));
    backbone="timm:convnext_small.dinov3_lvd1689m" — CNN из timm, признак = avg-pool.
    freeze_blocks: для ViT — число замороженных блоков, для ConvNeXt — замороженных стадий.
    """

    def __init__(self, backbone="facebook/dinov2-base", freeze_blocks=6, grad_ckpt=True):
        super().__init__()
        self.is_timm = backbone.startswith("timm:")
        if self.is_timm:
            import timm
            self.backbone = timm.create_model(backbone[5:], pretrained=True, num_classes=0)
            dim = self.backbone.num_features
            frozen = [self.backbone.stem] + list(self.backbone.stages[:freeze_blocks]) if freeze_blocks else []
            for m in frozen:
                for p in m.parameters():
                    p.requires_grad = False
            if grad_ckpt:
                self.backbone.set_grad_checkpointing(True)
        else:
            self.backbone = Dinov2Model.from_pretrained(backbone)
            dim = self.backbone.config.hidden_size * 2
            # ранние блоки заморожены: мало данных (1.2k ТС) и 4 ГБ видеопамяти
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for blk in self.backbone.encoder.layer[:freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False
            if grad_ckpt:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad = False  # BNNeck
        self.dim = dim

    def forward(self, x):
        if self.is_timm:
            feat = self.backbone(x)
        else:
            h = self.backbone(pixel_values=x).last_hidden_state
            feat = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        return feat, self.bn(feat)

    @torch.no_grad()
    def embed(self, x):
        return F.normalize(self.forward(x)[1], dim=1)


class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, s=30.0, m=0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, x, y):
        cos = F.linear(F.normalize(x), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        # без штрафа там, где theta + m выходит за pi (стандартный easy-margin-фикс)
        target = torch.where(theta + self.m < math.pi, target, cos - self.m * math.sin(self.m))
        onehot = F.one_hot(y, cos.size(1)).bool()
        return self.s * torch.where(onehot, target, cos)


def batch_hard_triplet(feat, y, margin=0.3):
    f = F.normalize(feat, dim=1)
    d = torch.cdist(f, f)
    same = y[:, None] == y[None]
    hardest_pos = (d * same).max(1).values
    hardest_neg = (d + same * 1e4).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()
