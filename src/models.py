import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
# set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

class simple_ann_model(nn.Module):
    def __init__(self, input_shape = 187):
        super(simple_ann, self).__init__()

        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x))

class cnn_ann_model(nn.Module):
    def __init__(self, input_shape = 187):
        super(cnn_ann_model, self).__init__()

        self.gru = nn.Conv2d(1, 32, 3, padding=1) # same 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # same 32
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(24)
        self.bn3 = nn.BatchNorm1d(60)
    
    def forward(self, signal):
        
        return torch.sigmoid(self.out(x))