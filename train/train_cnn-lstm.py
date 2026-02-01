import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.encoder import CNNEncoder
from models.decoder import RNNDecoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "mathwriting"

embedding_dim = 256
rnn_units = 512
VOCAB_SIZE = 62
BATCH_SIZE = 16
EPOCHS = 10
learning_rate = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = CNNEncoder(embedding_dim=256).to(DEVICE)
decoder = RNNDecoder(vocab_size=VOCAB_SIZE, embedding_dim=256, rnn_units=512).to(DEVICE)

images = torch.load(data_dir / "images_train.pt")  # (N, 1, 128, 128)
tokens = torch.load(data_dir / "tokens_train.pt")  # (N, seq_len)

dataset = TensorDataset(images, tokens)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate) 

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()

    total_loss = 0

    for imgs, seqs in loader:
        imgs = imgs.to(DEVICE)
        seqs = seqs.to(DEVICE)

        # teacher forcing, use ground truth tokens as input
        input_tokens = seqs[:, :-1]   # (B, seq_len-1)
        target_tokens = seqs[:, 1:]   # (B, seq_len-1)

        optimizer.zero_grad()  # clear previous gradients

        # encode images
        image_features = encoder(imgs)  # (B, 256)

        # decode sequences, pass encoder_features for initial hidden
        logits, _ = decoder(input_tokens, encoder_features=image_features)

        # compute loss
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            target_tokens.reshape(-1)
        )

        loss.backward()  # backpropagate
        optimizer.step()  # update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

