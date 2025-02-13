import numpy as np
import torch.nn as nn



class PolNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        logits = self.network(state)
        return logits

class ValNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)
