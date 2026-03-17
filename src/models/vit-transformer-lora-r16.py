import torch
from pickle import load
import sys
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class ViTLatexModelLoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, num_layers=6,
             lora_r=16, lora_alpha=32, lora_dropout=0.1,
             dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        target_modules = ["query", "key", "value", "dense", "fc1", "fc2"]
        
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",  # safe default for encoder-only usage
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        encoder_dim = self.encoder.config.hidden_size  # 768

        # Decoder embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, embed_dim) * 0.02)  # max seq length

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # linear projection to vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # project encoder dim to decoder embedding
        if encoder_dim != embed_dim:
            self.enc_proj = nn.Linear(encoder_dim, embed_dim)
        else:
            self.enc_proj = nn.Identity()

    def forward(self, images, input_tokens):
        """
        images: (B, 3, H, W)
        input_tokens: (B, T)
        """
        enc = self.encoder(pixel_values=images).last_hidden_state  # (B, N, D)
        enc = self.enc_proj(enc)  # (B, N, E)
    
        # Decoder embedding + positions
        emb = self.embedding(input_tokens) + self.pos_encoder[:, :input_tokens.size(1), :]  # (B, T, E)
    
        # Causal mask for decoder
        T = input_tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=images.device)
    
        # PAD mask (ignore padded tokens)
        PAD_ID = 0  
        tgt_key_padding_mask = (input_tokens == PAD_ID)  # True where padding
    
        # Transformer decoder
        dec_out = self.transformer_decoder(
            tgt=emb,
            memory=enc,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (B, T, E)
    
        # Project to vocab
        logits = self.fc_out(dec_out)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, image, max_len=150, sos_idx=1, eos_idx=2, beam_width=5):
        self.eval()
        device = image.device
        if image.ndim == 3: image = image.unsqueeze(0)
    
        # Encode image
        enc = self.encoder(pixel_values=image).last_hidden_state
        enc = self.enc_proj(enc)
    
        # Start with SOS
        beams = [(torch.tensor([sos_idx], device=device), 0.0)]
        completed = []
    
        for _ in range(max_len):
            all_candidates = []
    
            for seq, score in beams:
                if seq[-1].item() == eos_idx:
                    completed.append((seq, score))
                    continue
    
                T = seq.size(0)
                emb = self.embedding(seq.unsqueeze(0)) + self.pos_encoder[:, :T, :]
                mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
                out = self.transformer_decoder(tgt=emb, memory=enc, tgt_mask=mask)
                logits = self.fc_out(out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
    
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for logp, idx in zip(topk_log_probs, topk_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    all_candidates.append((new_seq, score + logp.item()))
    
            # Keep top beam_width sequences
            beams = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:beam_width]
            if all(s[-1].item() == eos_idx for s, _ in beams):
                break
    
        # Best sequence among completed and remaining beams
        all_final = completed + beams
        best_seq, _ = max(all_final, key=lambda x: x[1] / (len(x[0]) ** 0.7))
    
        # Strip SOS/EOS
        output = best_seq.tolist()
        if output[0] == sos_idx: output = output[1:]
        if output[-1] == eos_idx: output = output[:-1]
        return output