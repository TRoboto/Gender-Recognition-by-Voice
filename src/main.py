from data_utils import *
from train_utils import train_model
from models import * 
import torch.optim.lr_scheduler.StepLR

def main():
    # Load the dataset
    X, y = load_data()
    # Balance it
    Xb, yb = balance_dataset()
    # Split the data
    X_train, X_val, y_train, y_val = split_dataset(Xb, yb)
    # Get dataloaders
    dataloaders['train'] = get_dataloader(X_train, y_train)
    dataloaders['val'] = get_dataloader(X_val, y_val, 'val')
    # Define models
    net = simple_ann()
    # Define the loss fn, the optimizer and the scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # train the model
    train_model(net, criterion, optimizer, scheduler, dataloaders, 'simple_ann')


if __name__ == "__main__":
    main()