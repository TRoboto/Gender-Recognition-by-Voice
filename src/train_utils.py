import math
import sys
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, dataloaders, model_name, num_epochs=config.EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # switch to cuda if available
            model.to(device)

            batch_size = config.TRAIN_BATCH_SIZE if phase == 'train' else config.VALID_BATCH_SIZE
            datalength = len(dataloaders[phase].dataset)
            total_length = datalength // batch_size

            counter = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=total_length):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.float)
                # labels = labels.type(torch.FloatTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(labels.view_as(outputs)
                                              == torch.round(outputs))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datalength
            epoch_acc = running_corrects.double() / datalength

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save best model weights
    torch.save(best_model_wts, 'results/' + model_name +
               '_acc_{:.4f}'.format(best_acc.cpu().detach().numpy()) + ".pt")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_fn(model, test_loader):
    ''' Evaluate model on the test set 
    '''
    model.eval()
    model.to(device)
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            running_corrects += torch.sum(labels.view_as(outputs)
                                          == torch.round(outputs))

    print('Test set ACC: %.2f%%' % (running_corrects.cpu(
    ).detach().numpy() / len(test_loader.dataset) * 100))
