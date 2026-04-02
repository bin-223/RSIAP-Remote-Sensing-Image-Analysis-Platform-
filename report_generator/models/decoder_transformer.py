"""
Shared Transformer Decoder
适配:
- 共享 decoder
- 共享 vocab
- prompt token 方案
- 新版 encoder 输出的 visual_tokens

约定:
- encoder(images) 返回:
    {
        "visual_tokens": (B, N, d_model),
        "global_feat":   (B, d_model),
        "feat_map":      (B, d_model, Hf, Wf)
    }

- decoder.forward(...) 输入:
    visual_tokens: (B, N, d_model)
    captions:      (B, T)

其中 captions 在训练时应已经包含 prompt token，例如:
<START> <GLOBAL> ... tokens ... <END> <PAD> <PAD>

注意:
- prompt token 由词表和 dataset 负责构造
- decoder 本身不关心当前是哪个模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=256,
        pad_idx=0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)

    def generate_square_subsequent_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def build_tgt_key_padding_mask(self, captions):
        return captions.eq(self.pad_idx)

    def forward(self, visual_tokens, captions):
        device = captions.device
        seq_len = captions.size(1)

        tgt = self.embedding(captions) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        tgt_mask = self.generate_square_subsequent_mask(seq_len, device=device)
        tgt_key_padding_mask = self.build_tgt_key_padding_mask(captions)

        output = self.transformer_decoder(
            tgt=tgt,
            memory=visual_tokens,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc_out(output)

    @torch.no_grad()
    def generate_greedy(
        self,
        visual_tokens,
        prefix_tokens=None,
        start_token=1,
        end_token=2,
        max_len=100
    ):
        device = visual_tokens.device
        batch_size = visual_tokens.size(0)

        if prefix_tokens is None:
            generated = torch.full(
                (batch_size, 1),
                start_token,
                dtype=torch.long,
                device=device
            )
        else:
            generated = prefix_tokens.to(device)

        finished = generated[:, -1].eq(end_token)

        while generated.size(1) < max_len:
            logits = self.forward(visual_tokens, generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, end_token),
                next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(end_token)
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def generate_beam(
        self,
        visual_tokens,
        prefix_tokens=None,
        start_token=1,
        end_token=2,
        max_len=100,
        beam_size=3,
        length_penalty=0.7
    ):
        device = visual_tokens.device
        batch_size = visual_tokens.size(0)
        results = []

        for b in range(batch_size):
            memory = visual_tokens[b:b + 1]

            if prefix_tokens is None:
                init_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
            else:
                init_seq = prefix_tokens[b:b + 1].to(device)

            init_finished = init_seq[0, -1].item() == end_token
            beams = [(init_seq, 0.0, init_finished)]

            while beams[0][0].size(1) < max_len:
                all_candidates = []

                for seq, score, finished in beams:
                    if finished:
                        all_candidates.append((seq, score, finished))
                        continue

                    logits = self.forward(memory, seq)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        next_token = topk_indices[0, k].view(1, 1)
                        next_score = score + topk_log_probs[0, k].item()
                        next_seq = torch.cat([seq, next_token], dim=1)
                        next_finished = next_token.item() == end_token
                        all_candidates.append((next_seq, next_score, next_finished))

                def rank_fn(item):
                    seq, score, _ = item
                    seq_len = seq.size(1)
                    norm = ((5 + seq_len) / 6) ** length_penalty
                    return score / norm

                beams = sorted(all_candidates, key=rank_fn, reverse=True)[:beam_size]
                if all(finished for _, _, finished in beams):
                    break

            results.append(beams[0][0].squeeze(0))

        return results

    @torch.no_grad()
    def generate(
        self,
        visual_tokens,
        prefix_tokens=None,
        start_token=1,
        end_token=2,
        max_len=100,
        strategy="greedy",
        beam_size=3,
        length_penalty=0.7
    ):
        if strategy == "greedy":
            return self.generate_greedy(
                visual_tokens=visual_tokens,
                prefix_tokens=prefix_tokens,
                start_token=start_token,
                end_token=end_token,
                max_len=max_len
            )
        if strategy == "beam":
            return self.generate_beam(
                visual_tokens=visual_tokens,
                prefix_tokens=prefix_tokens,
                start_token=start_token,
                end_token=end_token,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty
            )
        raise ValueError(f"Unsupported generation strategy: {strategy}")
