
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

class CNNEncoder(Model):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2,2))
        
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2,2))
        
        self.conv3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2,2))
        
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(embedding_dim, activation='relu')  # final feature vector

    def call(self, x):
        x = self.conv1(x)  # each conv layer extracts complex features
        x = self.pool1(x)  # each pool layer reduces spatial dimensions, keeping important features
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)  # converts the final 2D feature maps to 1D vector of size (batch_size, features)
        x = self.fc(x)  # final dense layer to get desired embedding dimension
        return x
# example usage
if __name__ == "__main__":
    # array with random numbers between 0 and 1
    sample_input = np.random.rand(1, 128, 128, 1).astype(np.float32)  # shape (batch_size, height, width, channels)
    encoder = CNNEncoder()
    features = encoder(sample_input)  # forward pass, calls the call method
    print("Feature vector shape:", features.shape)  # (1, 256)