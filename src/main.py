from data_utils import *
from train_utils import train_model
from models import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import config
import itertools
import threading
import time
import sys


# for loading animation
done = False


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading data ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     \n')


def run_1d():
    # Load the dataset
    X, y = load_data()
    # Balance it
    Xb, yb = balance_dataset(X, y)
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(Xb, yb)
    # Normalize
    X_train = normalize(X_train)
    X_val = normalize(X_val, False)
    # Get dataloaders
    dataloaders = {}
    dataloaders['train'] = get_dataloader(X_train, y_train)
    dataloaders['val'] = get_dataloader(X_val, y_val, 'val')
    # Define models
    net = simple_ann_model()
    # Define the loss fn, the optimizer and the scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # train the model
    train_model(net, criterion, optimizer, scheduler,
                dataloaders, 'simple_ann_model')


def run_2d():
    global done
    # read dataframe
    X, y = load_2d_data()
    # stop loading animation
    done = True
    # Balance it
    Xb, yb = balance_dataset(X, y)
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(Xb, yb)
    # Get dataloaders
    dataloaders = {}
    dataloaders['train'] = get_dataloader(X_train, y_train)
    dataloaders['val'] = get_dataloader(X_val, y_val, 'val')
    # Define models
    net = rnn_model()
    # Define the loss fn, the optimizer and the scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # train the model
    train_model(net, criterion, optimizer, scheduler,
                dataloaders, 'rnn_model')


if __name__ == "__main__":
    t = threading.Thread(target=animate)
    t.start()
    run_2d()
