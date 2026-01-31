import tensorflow as tf
from tensorflow.keras import layers, Model

class RNNDecoder(Model):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=512):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)  # map token ids to embeddings
        self.lstm = layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(vocab_size)  # final output: logits for each token

    def call(self, x, hidden_state=None, encoder_output=None):
        x = self.embedding(x)  # turn token IDs into embeddings
        if hidden_state is None:
            output, state_h, state_c = self.lstm(x)
        else:
            output, state_h, state_c = self.lstm(x, initial_state=hidden_state)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        # state_h is the final hidden state of the current output and state_c is a memory to remember important info from previous tokens
        return logits, (state_h, state_c)  
decoder = RNNDecoder(vocab_size=62, embedding_dim=256, rnn_units=512)
sample_tokens = tf.constant([[1, 5, 3]])  # batch of 1 sequence
logits, state = decoder(sample_tokens)
print(logits.shape)  # (1, 3, 62)