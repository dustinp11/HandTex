import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):  # additive attention mechanism
    def __init__(self, enc_dim, dec_units):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, dec_units)  # to transform encoder features to same dimension as decoder hidden state
        self.W2 = nn.Linear(dec_units, dec_units)
        self.V = nn.Linear(dec_units, 1)  # to get attention scores for each encoder feature
    def forward(self, encoder_features, hidden):
        # hidden: (B, dec_units) -> (B, 1, dec_units) to broadcast
        hidden_with_time = hidden.unsqueeze(1)
        # score: (B, seq_len, 1)
        score = self.V(torch.tanh(self.W1(encoder_features) + self.W2(hidden_with_time)))
        # attention_weights: (B, seq_len, 1)
        attention_weights = F.softmax(score, dim=1)  # softmax over seq_len to get weights that sum to 1
        # context_vector: weighted sum over encoder features
        context_vector = torch.sum(attention_weights * encoder_features, dim=1)  # (B, enc_dim)
        # context_vector is the weighted average of encoder features
        return context_vector, attention_weights  # attention_weights show how much each encoder feature contributed to the context vector

