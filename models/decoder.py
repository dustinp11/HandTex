import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # map token ids to embeddings
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)  # input size is embedding_dim, output size is rnn_units
        self.enc_to_h = nn.Linear(embedding_dim, rnn_units)  # change encoder features to initial hidden state size
        self.fc = nn.Linear(rnn_units, vocab_size)  # final output: logits for each token, size vocab_size

    def forward(self, x, encoder_features, hidden=None):
        x = self.embedding(x)  # turn token IDs into embeddings

        if hidden is None:
            h0 = torch.tanh(self.enc_to_h(encoder_features)).unsqueeze(0)  # map encoder features to initial hidden state
            c0 = torch.zeros_like(h0)  # initial cell state
            hidden = (h0, c0)

        output, hidden = self.lstm(x, hidden)  # returns output for all time steps and final hidden state
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        # state_h is the final hidden state of the current output and state_c is a memory to remember important info from previous tokens
        return logits, hidden
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