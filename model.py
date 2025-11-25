import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

class CLCNet(nn.Module):
    def __init__(self,input_dim=67500, shared_dim=[4096, 2048, 1024]):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim[0]),
            nn.Dropout(0.3),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim[1]),
            nn.Dropout(0.3),
            nn.Linear(shared_dim[1], shared_dim[2]))
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]))
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1))
        self.aux_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[0]))
    def forward(self, x):
        shared_features = self.shared_layer(x)
        shared_features = shared_features + self.shared_layer_1(x)
        main_output = self.main_task_layer(shared_features)
        aux_output = self.aux_task_layer(shared_features)
        aux_output = F.normalize(aux_output, p=2, dim=1)
        return main_output, aux_output
