import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

class BahdanauAttention(nn.Module):
    """
    Additive attention:
      score(h, e_i) = v^T tanh(W_h h + W_e e_i)
    """
    def __init__(self, hidden_dim, encoder_dim, attn_dim=None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_e = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, enc_mem, enc_mask=None):
        """
        h:        (B, H)
        enc_mem:  (B, N, D)
        enc_mask: (B, N) with 1 for valid, 0 for pad (optional)

        returns:
          context: (B, D)
          alpha:   (B, N)
        """
        # (B, 1, A) + (B, N, A) -> (B, N, A)
        energy = torch.tanh(self.W_h(h).unsqueeze(1) + self.W_e(enc_mem))
        scores = self.v(energy).squeeze(-1)  # (B, N)

        if enc_mask is not None:
            scores = scores.masked_fill(enc_mask == 0, -1e9)

        alpha = F.softmax(scores, dim=-1)    # (B, N)
        context = torch.bmm(alpha.unsqueeze(1), enc_mem).squeeze(1)  # (B, D)
        return context, alpha


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = BahdanauAttention(hidden_dim, encoder_dim, attn_dim=hidden_dim)

        # Step-wise LSTM so we can inject attention context every timestep
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Optional: init decoder hidden state from encoder (pooled)
        self.enc_to_h = nn.Linear(encoder_dim, hidden_dim)
        self.enc_to_c = nn.Linear(encoder_dim, hidden_dim)

    def init_hidden(self, enc_mem):
        """
        enc_mem: (B, N, D)
        """
        pooled = enc_mem.mean(dim=1)  # (B, D)
        h0 = torch.tanh(self.enc_to_h(pooled))
        c0 = torch.tanh(self.enc_to_c(pooled))
        return (h0, c0)

    def forward(self, x, enc_mem=None, hidden_state=None, enc_mask=None):
        """
        x:       (B, T) token ids (teacher forcing input sequence)
        enc_mem: (B, N, D) ViT patch tokens (memory)
        hidden_state: (h, c) each (B, H) or None

        returns:
          logits: (B, T, vocab)
          hidden: (h, c)
        """
        B, T = x.shape
        emb = self.dropout(self.embedding(x))  # (B, T, E)

        if hidden_state is None:
            if enc_mem is None:
                # fallback: start from zeros
                h = torch.zeros(B, self.lstm_cell.hidden_size, device=x.device)
                c = torch.zeros_like(h)
            else:
                h, c = self.init_hidden(enc_mem)
        else:
            h, c = hidden_state

        logits_steps = []

        for t in range(T):
            if enc_mem is not None:
                context, _ = self.attn(h, enc_mem, enc_mask=enc_mask)  # (B, D)
            else:
                # no encoder: zero context
                context = torch.zeros(B, self.enc_to_h.in_features, device=x.device)

            lstm_in = torch.cat([emb[:, t, :], context], dim=-1)  # (B, E+D)
            h, c = self.lstm_cell(lstm_in, (h, c))
            logits_steps.append(self.fc(h).unsqueeze(1))  # (B, 1, vocab)

        logits = torch.cat(logits_steps, dim=1)  # (B, T, vocab)
        return logits, (h, c)


#============================================================================

class ViTLatexModelLoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")

        # IMPORTANT: do NOT freeze everything after LoRA; LoRA will keep base frozen + adapters trainable.
        # (Freezing before LoRA is fine, but not necessary.)

        # Choose target modules after you inspect names.
        # Start with common ViT block names:
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
        self.decoder = AttnDecoder(vocab_size, embed_dim, hidden_dim, encoder_dim)

    def forward(self, images, input_tokens):
        enc = self.encoder(pixel_values=images).last_hidden_state
        enc_mem = enc[:, 1:, :]  # (B, N, D)
        logits, _ = self.decoder(input_tokens, enc_mem=enc_mem)
        return logits

    @torch.no_grad()
    def generate(self, image, max_len=150, sos_idx=1, eos_idx=2):
        self.eval()

        # image: (1, 3, H, W) already normalized in your eval loop
        enc = self.encoder(pixel_values=image).last_hidden_state
        enc_mem = enc[:, 1:, :]  # (1, N, D)

        token = torch.tensor([[sos_idx]], device=image.device)
        output_tokens = []
        hidden = None

        for _ in range(max_len):
            logits, hidden = self.decoder(token, enc_mem=enc_mem, hidden_state=hidden)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)

            if next_token.item() == eos_idx:
                break

            output_tokens.append(next_token.item())
            token = next_token

        return output_tokens
    

