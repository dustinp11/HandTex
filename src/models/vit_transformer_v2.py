import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class ViTLatexModelLoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=8, num_layers=6,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        target_modules = ["query", "key", "value", "dense", "fc1", "fc2"]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        encoder_dim = self.encoder.config.hidden_size  # 768

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        if encoder_dim != embed_dim:
            self.enc_proj = nn.Linear(encoder_dim, embed_dim)
        else:
            self.enc_proj = nn.Identity()

    def forward(self, images, input_tokens):
        enc = self.encoder(pixel_values=images).last_hidden_state
        enc = self.enc_proj(enc)
        emb = self.embedding(input_tokens) + self.pos_encoder[:, :input_tokens.size(1), :]
        T = input_tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(images.device)
        dec_out = self.transformer_decoder(tgt=emb, memory=enc, tgt_mask=causal_mask)
        return self.fc_out(dec_out)

    @torch.no_grad()
    def generate(self, image, max_len=150, sos_idx=1, eos_idx=2):
        self.eval()
        device = image.device
        enc = self.encoder(pixel_values=image).last_hidden_state
        enc = self.enc_proj(enc)
        tokens = torch.tensor([[sos_idx]], device=device)
        for _ in range(max_len):
            emb = self.embedding(tokens) + self.pos_encoder[:, :tokens.size(1), :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(device)
            out = self.transformer_decoder(tgt=emb, memory=enc, tgt_mask=causal_mask)
            next_token = self.fc_out(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            if next_token.item() == eos_idx:
                break
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens[0, 1:].tolist()