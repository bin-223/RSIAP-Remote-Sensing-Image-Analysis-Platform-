import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    V2 Image Encoder
    仅支持:
        - resnet50
        - resnet101

    输出:
        {
            "visual_tokens": (B, N, d_model),
            "global_feat":   (B, d_model),
            "feat_map":      (B, d_model, Hf, Wf)
        }

    说明:
        - visual_tokens: 给后续 Transformer decoder 做 cross-attention
        - global_feat: 可给分类头、控制模块、全局引导使用
        - feat_map: 预留给后续可能的空间模块/可视化/检测头
    """

    SUPPORTED_ENCODERS = ["resnet50", "resnet101"]

    def __init__(
        self,
        encoder_name="resnet101",
        d_model=512,
        dropout=0.1,
        pretrained=True,
        train_backbone=True
    ):
        super().__init__()

        if encoder_name not in self.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. "
                f"Only {self.SUPPORTED_ENCODERS} are supported."
            )

        self.encoder_name = encoder_name
        self.d_model = d_model

        if encoder_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            cnn = models.resnet50(weights=weights)
            backbone_dim = 2048

        elif encoder_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            cnn = models.resnet101(weights=weights)
            backbone_dim = 2048

        self.backbone = nn.Sequential(*list(cnn.children())[:-2])

        self.feature_proj = nn.Conv2d(
            in_channels=backbone_dim,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )

        self.feature_bn = nn.BatchNorm2d(d_model)
        self.feature_act = nn.ReLU(inplace=True)
        self.feature_dropout = nn.Dropout2d(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.token_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)

        if not train_backbone:
            self.freeze_backbone()

    def forward(self, images):
        feat = self.backbone(images)
        feat = self.feature_proj(feat)
        feat = self.feature_bn(feat)
        feat = self.feature_act(feat)
        feat = self.feature_dropout(feat)

        feat_map = feat
        visual_tokens = feat_map.flatten(2).transpose(1, 2)
        visual_tokens = self.token_norm(visual_tokens)

        global_feat = self.global_pool(feat_map).flatten(1)
        global_feat = self.global_norm(global_feat)

        return {
            "visual_tokens": visual_tokens,
            "global_feat": global_feat,
            "feat_map": feat_map
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
