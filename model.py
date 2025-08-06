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
    
class CLCNet_no_con(nn.Module):
    def __init__(self, input_dim=67500, shared_dim=[4096,2048,1024]):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[2]))
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]))
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1))  
    def forward(self, x_cat):
        shared_features = self.shared_layer(x_cat)
        shared_features = shared_features + self.shared_layer_1(x_cat)
        main_output = self.main_task_layer(shared_features)
        return main_output

class DeepGS(nn.Module):
    def __init__(self,dim=67500):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        return x
        
class DNNGP(nn.Module):
    def __init__(self, input_dim):
        super(DNNGP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.bn = nn.BatchNorm1d(num_features=64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x) 
        return x
