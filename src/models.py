import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


def simple_ann(nn.Module):
    def __init__(self, input_shape = 187):
        super(simple_ann, self).__init__()

        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
    
    def forward(self, x):

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn1(self.fc2(x))))
        x = self.dropout(F.relu(self.bn2(self.fc3(x))))
        x = self.dropout(F.relu(self.bn2(self.fc4(x))))
        x = self.dropout(F.relu(self.fc5(x)))
        out = F.sigmoid(self.out(x))

        return out