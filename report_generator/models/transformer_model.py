"""
完整的图像描述模型（Encoder + Shared Transformer Decoder）

适配方案:
- encoder: ResNet50 / ResNet101
- decoder: 共享 Transformer Decoder
- vocab: 共享 vocab
- module control: prompt token
- generation: 分别生成四段
"""

import torch
import torch.nn as nn

from models.encoder import ImageEncoder
from models.decoder_transformer import TransformerDecoder


class ImageCaptioningTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, max_visual_tokens=256):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pos_embed = nn.Embedding(max_visual_tokens, encoder.d_model)

    def encode(self, images):
        encoder_outputs = self.encoder(images)
        visual_tokens = encoder_outputs["visual_tokens"]
        _, num_tokens, _ = visual_tokens.shape
        positions = torch.arange(num_tokens, device=visual_tokens.device)
        pos_embedding = self.pos_embed(positions)
        encoder_outputs["visual_tokens"] = visual_tokens + pos_embedding.unsqueeze(0)
        return encoder_outputs

    def decode(self, visual_tokens, captions):
        return self.decoder(visual_tokens, captions)

    def forward(self, images, captions):
        encoder_outputs = self.encode(images)
        return self.decoder(encoder_outputs["visual_tokens"], captions)

    @torch.no_grad()
    def generate(
        self,
        images,
        prefix_tokens=None,
        start_token=1,
        end_token=2,
        max_len=100,
        strategy="greedy",
        beam_size=3,
        length_penalty=0.7
    ):
        encoder_outputs = self.encode(images)
        visual_tokens = encoder_outputs["visual_tokens"]
        return self.decoder.generate(
            visual_tokens=visual_tokens,
            prefix_tokens=prefix_tokens,
            start_token=start_token,
            end_token=end_token,
            max_len=max_len,
            strategy=strategy,
            beam_size=beam_size,
            length_penalty=length_penalty
        )

    @torch.no_grad()
    def generate_module(
        self,
        images,
        module_token_id,
        start_token=1,
        end_token=2,
        max_len=100,
        strategy="greedy",
        beam_size=3,
        length_penalty=0.7
    ):
        device = images.device
        batch_size = images.size(0)
        prefix_tokens = torch.tensor(
            [[start_token, module_token_id]] * batch_size,
            dtype=torch.long,
            device=device
        )
        return self.generate(
            images=images,
            prefix_tokens=prefix_tokens,
            start_token=start_token,
            end_token=end_token,
            max_len=max_len,
            strategy=strategy,
            beam_size=beam_size,
            length_penalty=length_penalty
        )

    @torch.no_grad()
    def generate_all_modules(
        self,
        images,
        module_token_ids,
        start_token=1,
        end_token=2,
        max_len=100,
        strategy="greedy",
        beam_size=3,
        length_penalty=0.7
    ):
        results = {}
        mapping = {
            "global": module_token_ids["global"],
            "detail": module_token_ids["detail"],
            "abnormal": module_token_ids["abnormal"],
            "conclusion": module_token_ids["conclusion"],
        }

        for key, module_token_id in mapping.items():
            results[key] = self.generate_module(
                images=images,
                module_token_id=module_token_id,
                start_token=start_token,
                end_token=end_token,
                max_len=max_len,
                strategy=strategy,
                beam_size=beam_size,
                length_penalty=length_penalty
            )

        return results


def build_image_captioning_transformer(
    vocab_size,
    encoder_name="resnet101",
    d_model=512,
    nhead=8,
    num_decoder_layers=4,
    dim_feedforward=2048,
    dropout=0.1,
    max_len=256,
    pad_idx=0,
    pretrained_encoder=True,
    train_backbone=True
):
    encoder = ImageEncoder(
        encoder_name=encoder_name,
        d_model=d_model,
        dropout=dropout,
        pretrained=pretrained_encoder,
        train_backbone=train_backbone
    )

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_idx=pad_idx
    )

    return ImageCaptioningTransformer(encoder, decoder, max_visual_tokens=max_len)
