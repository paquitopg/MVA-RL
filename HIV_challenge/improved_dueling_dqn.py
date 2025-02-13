import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedDuelingNetwork(nn.Module):
    def __init__(self, state_size=6, action_size=4, hdim=128, hdim2=64):
        super(ImprovedDuelingNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Layer Normalization for better training stability
        self.ln_input = nn.LayerNorm(state_size)
        
        # Shared feature extractor for better representation learning
        self.features = nn.Sequential(
            nn.Linear(state_size, hdim),
            nn.ReLU(),
            nn.LayerNorm(hdim),
            nn.Dropout(0.1),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.LayerNorm(hdim),
            nn.Dropout(0.1)
        )
        
        # Advantage stream with reduced complexity
        self.advantage = nn.Sequential(
            nn.Linear(hdim, hdim2),
            nn.ReLU(),
            nn.LayerNorm(hdim2),
            nn.Dropout(0.1),
            nn.Linear(hdim2, action_size)
        )
        
        # Value stream with reduced complexity
        self.value = nn.Sequential(
            nn.Linear(hdim, hdim2),
            nn.ReLU(),
            nn.LayerNorm(hdim2),
            nn.Dropout(0.1),
            nn.Linear(hdim2, 1)  # Changed to output a single value
        )
        
        # Initialize weights using Xavier/Glorot initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.ln_input(x)
        features = self.features(x)
        
        advantage = self.advantage(features)
        value = self.value(features)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
