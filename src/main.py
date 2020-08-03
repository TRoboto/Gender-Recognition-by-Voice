import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

import config
from data_utils import *
from models import *
from train_utils import *
from utils import plot_performance


def train_ann():
    # Load the dataset
    X, y = load_data()
    # Balance it
    Xb, yb = balance_dataset(X, y)
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(Xb, yb)
    # Normalize
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    # Get dataloaders
    dataloaders = {}
    dataloaders['train'] = get_dataloader(X_train, y_train)
    dataloaders['val'] = get_dataloader(X_val, y_val, 'val')
    # Define models
    model = ann_model()
    if os.path.isfile('results/ann_model.pt'):
        model.load_state_dict(torch.load('results/ann_model.pt'))
    else:
        # Define the loss fn, the optimizer and the scheduler
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # train the model
        train_model(model, criterion, optimizer, scheduler,
                    dataloaders, 'ann_model')

    # Load the test dataset
    X_test, y_test = load_test_data()
    # # Normalize
    X_test = normalize(X_test)
    # Get dataloader
    test_loader = get_dataloader(X_test, y_test)
    # evaluate
    eval_fn(model, test_loader)


def train_lazy():
    # Load the dataset
    X, y = load_data()
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(X, y)
    # # Normalize
    X_train = normalize(X_train)
    X_val = normalize(X_val)

    # uncomment to check the performance of the 25 models
    # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    # # fit
    # scores,_ = clf.fit(X_train, X_val, y_train, y_val)
    # # print
    # print(scores)

    # Final model
    # check if model exist
    if os.path.isfile(config.MODEL_PATH):
        model = XGBClassifier()
        model.load_model(config.MODEL_PATH)
    else:
        model = XGBClassifier()
        model.fit(X_train, y_train, eval_metric="error", eval_set=[
                  (X_train, y_train), (X_val, y_val)], verbose=True)
        # save model
        model.save_model(config.MODEL_PATH)
    # performance on train set
    y_pred = model.predict(X_train)
    # evaluate predictions
    print_performance(y_train, y_pred, 'train')

    # performance on val set
    y_pred = model.predict(X_val)
    # evaluate predictions
    print_performance(y_val, y_pred, 'val')

    # Load the test dataset
    X_test, y_test = load_test_data()
    # # Normalize
    X_test = normalize(X_test)
    # get prediction
    y_pred = model.predict(X_test)
    # evaluate predictions
    print_performance(y_test, y_pred, 'test')
    # print
    plot_performance(model)


def print_performance(y_true, y_pred, name):
    predictions = [round(value) for value in y_pred]
    perfs = [accuracy_score, balanced_accuracy_score,
             f1_score, roc_auc_score]
    for perf in perfs:
        print(perf.__name__ + f' on {name} set:' +
              '%.2f%%' % (perf(y_true, predictions) * 100.0))


if __name__ == "__main__":
    train_lazy()
