
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16, 32, 64], activation='relu', dropout=0.2):
        super(MLPAutoencoder, self).__init__()
        self.input_dim = input_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(self.activation)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for dim in hidden_dims[len(hidden_dims)//2 + 1:]:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(self.activation)
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            mse = F.mse_loss(recon, x, reduction='none')
            return mse.mean(dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
