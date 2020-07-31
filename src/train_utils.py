import math
import sys
import time
import torch
import numpy as np

def train_model(model, optimizer, data_loader, device, criterion, epochs = 10, print_freq = 1):
    model.train()

    model.to(device)
    for ep in range(1, epochs + 1):

        running_loss = 0.0
        for features, targets in data_loader:
            features = list(feat.to(device) for feat in features)
            targets = targets.to(device)
            
            outs = model(features)

            loss = criterion(outs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print loss statistics
            if ep % print_freq == 0:
                print('Epoch: {:>4}/{:<4} Avg. Loss: {}\n'.format(
                    ep, epochs, np.average(running_loss)))
                running_loss = 0
                
    print('Finished Training')