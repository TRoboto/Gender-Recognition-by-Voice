import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import config

# set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


class simple_ann_model(nn.Module):
    def __init__(self, input_shape=187):
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
    def __init__(self, input_shape=config.MAX_LENGTH):
        super(cnn_ann_model, self).__init__()

        self.conv1 = nn.Conv1d(input_shape, 32, 3, padding=1)  # same 64
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)  # same 32

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.out = nn.Linear(64, 1)

        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, signal):
        x = F.relu(self.conv1(signal))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(self.out(x))


class rnn_model(nn.Module):
    def __init__(self, input_dim=config.WINDOW_SIZE, hidden_dim = 512):
        super(rnn_model, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, signal):

        lstm_out, _ = self.lstm(signal)
        lstm_out = lstm_out[:, -1]
        out = self.dense(lstm_out)
        return torch.sigmoid(out)
