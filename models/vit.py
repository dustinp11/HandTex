import torch
import torch.nn as nn
from transformers import ViTModel


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.enc_to_h = nn.Linear(encoder_dim, hidden_dim)  # project encoder features to hidden size

    def forward(self, x, encoder_features=None, hidden_state=None):
        """
        x: token indices (batch, seq_len)
        encoder_features: encoder output (batch, encoder_dim) - used to init hidden state
        hidden_state: tuple of (h, c) for continuing generation
        """
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        if hidden_state is None:
            if encoder_features is not None:
                h0 = torch.tanh(self.enc_to_h(encoder_features)).unsqueeze(0)  # (1, batch, hidden_dim)
                c0 = torch.zeros_like(h0)
                output, hidden = self.lstm(x, (h0, c0))
            else:
                output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden_state)

        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden


class ViTLatexModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size  # 768
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, encoder_dim)

    def forward(self, images, targets):
        encoder_out = self.encoder(images).last_hidden_state[:, 0, :]  # (batch, 768)
        logits, _ = self.decoder(targets, encoder_features=encoder_out)
        return logits

    @torch.no_grad()
    def generate(self, image, max_len=100, sos_idx=1, eos_idx=2):
        """
        Autoregressive generation for inference.
        """
        self.eval()
        encoder_out = self.encoder(image).last_hidden_state[:, 0, :]  # (1, 768)

        token = torch.tensor([[sos_idx]], device=image.device)
        output_tokens = []
        hidden = None

        for i in range(max_len):
            if i == 0:
                logits, hidden = self.decoder(token, encoder_features=encoder_out)
            else:
                logits, hidden = self.decoder(token, hidden_state=hidden)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy

            if next_token.item() == eos_idx:
                break

            output_tokens.append(next_token.item())
            token = next_token

        return output_tokens

    def predict(self, image):
        # Placeholder for API
        return "x^2"