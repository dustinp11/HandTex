import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # map token ids to embeddings
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)  # input size is embedding_dim, output size is rnn_units
        self.fc = nn.Linear(rnn_units, vocab_size)  # final output: logits for each token, size vocab_size
        self.enc_to_h = nn.Linear(embedding_dim, rnn_units)  # change encoder features to initial hidden state size

    def forward(self, x, encoder_features = None,hidden_state=None):
        x = self.embedding(x)  # turn token IDs into embeddings
        if hidden_state is None:
            if encoder_features is not None:  # use encoder features to initialize hidden state
                h0 = torch.tanh(self.enc_to_h(encoder_features)).unsqueeze(0)  # map encoder features to initial hidden state
                c0 = torch.zeros_like(h0)  # initial cell state
                output, hidden = self.lstm(x, (h0, c0))
            else:
                output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        # state_h is the final hidden state of the current output and state_c is a memory to remember important info from previous tokens
        return logits, hidden
if __name__ == "__main__":
    decoder = RNNDecoder(vocab_size=62)

    sample_tokens = torch.tensor([[1, 5, 3]])
    logits, hidden = decoder(sample_tokens)

    print("Logits shape:", logits.shape)