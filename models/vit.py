import torch
import torch.nn as nn
from transformers import ViTModel


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_out, targets):
        # TODO: implement forward pass with teacher forcing
        pass


class ViTLatexModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = Decoder(vocab_size, hidden_dim=self.encoder.config.hidden_size)

    def forward(self, images, targets):
        encoder_out = self.encoder(images).last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.decoder(encoder_out, targets)
        return logits

    @torch.no_grad()
    def generate(self, image, max_len=100, sos_idx=1, eos_idx=2):
        """
        Autoregressive generation for inference.
        """
        self.eval()
        encoder_out = self.encoder(image).last_hidden_state[:, 0, :]  # (1, 768)

        h = encoder_out.unsqueeze(0).repeat(2, 1, 1)
        c = torch.zeros_like(h)

        token = torch.tensor([[sos_idx]], device=image.device)
        output_tokens = []

        for _ in range(max_len):
            emb = self.decoder.embedding(token)  # (1, 1, embed_dim)
            out, (h, c) = self.decoder.lstm(emb, (h, c))
            logits = self.decoder.fc(out)  # (1, 1, vocab_size)
            next_token = logits.argmax(dim=-1)  # greedy

            if next_token.item() == eos_idx:
                break

            output_tokens.append(next_token.item())
            token = next_token

        return output_tokens

    def predict(self, image):
        # Placeholder for API
        return "x^2"