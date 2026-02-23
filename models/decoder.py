import torch
import torch.nn as nn
from attention import Attention

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=512, enc_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # map token ids to embeddings
        self.attention = Attention(enc_dim, rnn_units)  # attention mechanism to focus on relevant encoder features
        self.lstm = nn.LSTM(embedding_dim + enc_dim, rnn_units, batch_first=True)  # embedding_dim + enc_dim so that it can see the image features at each time step
        self.enc_to_h = nn.Linear(enc_dim, rnn_units)  # change encoder features to initial hidden state size
        self.fc = nn.Linear(rnn_units, vocab_size)  # final output: logits for each token, size vocab_size

    def forward(self, x, encoder_features, hidden=None):
        B, T = x.shape
        x_embed = self.embedding(x)  # turn token IDs into embeddings
        outputs = []
        if hidden is None:
            h0 = torch.tanh(self.enc_to_h(encoder_features)).unsqueeze(0)  # map encoder features to initial hidden state
            c0 = torch.zeros_like(h0)  # initial cell state
            hidden = (h0, c0)

        h_t, c_t = hidden
        for t in range(T):
            x_t = x_embed[:, t, :]  # (B, embedding_dim)
            context, attn_weights = self.attention(encoder_features, h_t.squeeze(0))
            lstm_input = torch.cat([x_t, context], dim=-1).unsqueeze(1)
            out, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            outputs.append(out)
        output = torch.cat(outputs, dim=1)  # (B, T, rnn_units)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        # state_h is the final hidden state of the current output and state_c is a memory to remember important info from previous tokens
        return logits, (h_t, c_t)
if __name__ == "__main__":
    sample_tokens = torch.tensor([[1, 5, 3]])  # token IDs (B=1, seq_len=3)

    encoder_features = torch.rand(1, 256)  # (B=1, embedding_dim)

    decoder = RNNDecoder(vocab_size=62)

    logits, hidden = decoder(sample_tokens, encoder_features)

    print("Logits shape:", logits.shape)  # (B, seq_len, vocab_size)
    print("Hidden state shape:", hidden[0].shape)  # state_h: (num_layers, B, rnn_units)
    print("Cell state shape:", hidden[1].shape)  # state_c: (num_layers, B, rnn_units)
    print(logits)
    print(logits.argmax(-1))

    