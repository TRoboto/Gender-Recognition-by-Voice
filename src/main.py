import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from lazypredict.Supervised import LazyClassifier

import config
from data_utils import *
from models import *
from train_utils import train_model


def train_ann():
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
    net = ann_model()
    # Define the loss fn, the optimizer and the scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # train the model
    train_model(net, criterion, optimizer, scheduler,
                dataloaders, 'ann_model')

def train_lazy():
    # Load the dataset
    X, y = load_data()
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(X, y)
    # # Normalize
    # X_train = normalize(X_train)
    # X_val = normalize(X_val, False)
    # define classifier
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    # fit
    models,predictions = clf.fit(X_train, X_val, y_train, y_val)
    # print
    print(models)

if __name__ == "__main__":
    train_lazy()
