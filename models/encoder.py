import torch
import torch.nn as nn
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # grayscale input
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*16*16, embedding_dim)  # final feature vector to embedding_dim
        self.ln = nn.LayerNorm(embedding_dim)  # normalize features

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # each conv layer extracts complex features
        x = self.pool1(x)  # each pool layer reduces spatial dimensions, keeping important features
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.flatten(x)  # converts the final 2D feature maps to 1D vector of size (batch_size, features)
        x = torch.relu(self.fc(x))
        x = self.ln(x)  # normalize feature vector
        return x  
# example usage
if __name__ == "__main__":
    # array with random numbers between 0 and 1
    sample_input = np.random.rand(1, 1, 128, 128).astype(np.float32)  # shape (batch_size, channels, height, width)
    sample_input = torch.tensor(sample_input)
    encoder = CNNEncoder()
    features = encoder(sample_input)  # forward pass, calls the call method
    print("Feature vector shape:", features.shape)  # (1, 256)