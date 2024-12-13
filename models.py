from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

def build_cnn(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Conv2d(layers_dims[i], layers_dims[i + 1], kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Conv2d(layers_dims[-2], layers_dims[-1], kernel_size=3, stride=2, padding=1))
    return nn.Sequential(*layers)

class Encoder(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        # self.cnn = build_cnn([2, 32, 64, 128, 256, embedding_dim])
        self.fc = nn.Linear(embedding_dim * 5 * 5, embedding_dim)
        # self.fc = nn.Linear(embedding_dim * 3 * 3, embedding_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class Predictor(torch.nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.fc = build_mlp([embedding_dim + action_dim, embedding_dim, embedding_dim])

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

class JEPA(torch.nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2, momentum=0.99):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.predictor = Predictor(embedding_dim=embedding_dim, action_dim=action_dim)
        self.target_encoder = Encoder(embedding_dim=embedding_dim)
        self.repr_dim = embedding_dim
        self.momentum = momentum

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, states, actions):
        b, t, c, h, w = states.size()
        _, timesteps, _ = actions.size()
        states_reshaped = states.view(-1, c, h, w)
        encoded_states = self.encoder(states_reshaped)
        encoded_states = encoded_states.view(b, t, -1)
        predicted_states = [encoded_states[:, 0]]

        if t == 1:
            pred_state = encoded_states[:, 0]
            for ts in range(timesteps):
                pred_state = self.predictor(pred_state, actions[:, ts])
                predicted_states.append(pred_state)
        else:
            for ts in range(timesteps):
                pred_state = self.predictor(encoded_states[:, ts], actions[:, ts])
                predicted_states.append(pred_state)
        
        return torch.stack(predicted_states, dim=1)
    
    @torch.no_grad()
    def update_target_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output