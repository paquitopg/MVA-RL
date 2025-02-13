import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

class DDeepDuelingNetwork(nn.Module):
    def __init__(self, hdim, state_size = 6, action_size = 4):
        self.state_size = state_size
        self.action_size = action_size
        super(DDeepDuelingNetwork, self).__init__()

        self.advantage = nn.Sequential(
            nn.Linear(state_size, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(state_size, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_size)
        )
    
    def forward(self, x):
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()