import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, encoded_dim)

    def forward(self, x):
        # x: (batch, 1, H, W)
        features = self.cnn(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features  # (batch, encoded_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_out, targets):
        # encoder_out: (batch, hidden_dim) - used to init hidden state
        # targets: (batch, seq_len) - teacher forcing input
        embeddings = self.embedding(targets)  # (batch, seq_len, embed_dim)

        h0 = encoder_out.unsqueeze(0)  # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)

        outputs, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
        return logits


class CNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, encoded_dim=256, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(encoded_dim=hidden_dim)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, images, targets):
        features = self.encoder(images)
        logits = self.decoder(features, targets)
        return logits

    @torch.no_grad()
    def generate(self, image, max_len=100, sos_idx=1, eos_idx=2):
        # Greedy decoding for inference
        self.eval()
        features = self.encoder(image)  # (1, hidden_dim)

        h = features.unsqueeze(0)
        c = torch.zeros_like(h)

        token = torch.tensor([[sos_idx]], device=image.device)
        output_tokens = []

        for _ in range(max_len):
            emb = self.decoder.embedding(token)
            out, (h, c) = self.decoder.lstm(emb, (h, c))
            logits = self.decoder.fc(out)
            next_token = logits.argmax(dim=-1)

            if next_token.item() == eos_idx:
                break

            output_tokens.append(next_token.item())
            token = next_token

        return output_tokens