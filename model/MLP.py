
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

OneLayer = "1_layer"
TwoLayer = "2_layer"

class MLP(nn.Module):
    def __init__(self, 
                 dim, 
                 projection_size, 
                 hidden_size=4096, 
                 num_layer=TwoLayer):
        super().__init__()
        
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)