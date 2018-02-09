import os
import logging
from tqdm import tqdm

import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from . import utils

cuda_is_available = torch.cuda.is_available()

INIT_CYCLIC_LR = 2e-5


def cyclic_lr(epoch, init_lr, num_epochs_per_cycle=10, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    return lr


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(torch.autograd.Variable(x.cuda(async=True), volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


class Tracker(object):
    def __init__(self):
        self._l1 = self._l2 = self._l3 = float('+inf')
        self._not_updated_for = 0

    def should_save(self, new_loss):
        if new_loss < self._l1:
            self._l3 = self._l2
            self._l2 = self._l1
            self._l1 = new_loss
            self._not_updated_for = 0
            return True
        self._not_updated_for += 1
        if new_loss > self._l3:
            return False
        if new_loss < self._l2:
            self._l3 = self._l2
            self._l2 = new_loss
        else:
            self._l3 = new_loss
        return True

    def not_updated_for(self):
        return self._not_updated_for


def train_and_validate(
        train_data_loader,
        valid_loaders,
        model,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        start_epoch,
        best_val_loss,
        experiment_name,
):
    checkpoint_dir = os.getcwd() + '/models/{}'.format(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tracker = Tracker()
    cyclic_lr_started = False
    valid_losses = []

    for epoch in range(start_epoch, epochs + 1):
        train(
            train_data_loader,
            model,
            optimizer,
            loss_fn,
            epoch,
        )
        val_loss = validate(
            valid_loaders,
            model,
            loss_fn,
        )
        valid_losses.append(val_loss)

        if tracker.should_save(val_loss):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            },
                '{experiment_name}_{epoch}_{val_loss}.pth'.format(experiment_name=experiment_name, epoch=epoch,
                                                                  val_loss=val_loss),
                checkpoint_dir,
            )
        if tracker.not_updated_for() >= 6 and not cyclic_lr_started:
            cyclic_lr_started = True
            optimizer = torch.optim.SGD(model.parameters(), lr=INIT_CYCLIC_LR)
            print(logging.info('Starting cyclic lr'))
        if not cyclic_lr_started:
            scheduler.step(val_loss, epoch)
        else:
            for param_group in optimizer.param_groups:
                new_lr = cyclic_lr(epoch, init_lr=INIT_CYCLIC_LR)
                param_group['lr'] = new_lr
    return model


def train(train_loader, model, optimizer, criterion, epoch):
    losses = []

    model.train()
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    logging.info('Epoch: {} | lrs: {}'.format(epoch, lrs))
    for i, (inputs, O, targets) in enumerate(train_loader):
        inputs, O, targets = variable(inputs), variable(O), variable(targets)
        #print(targets)
        #print(list(targets))
        outputs = model(inputs, O)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        batch_size = inputs.size(0)
        (batch_size * loss).backward()
        optimizer.step()

        losses.append(loss.data[0])

        if (i + 1) % 100 == 0:
            logging.info('Step: {}, train_loss: {}'.format(i + 1, np.mean(losses[-100:])))

    train_loss = np.mean(losses)
    logging.info('train_loss: {}'.format(train_loss))


def validate(val_loaders, model, criterion):
    valid_loss = dict()
    valid_accuracy = dict()
    for mode in ('unalt', 'manip'):
        accuracy_scores = []
        losses = []

        model.eval()

        for i, (inputs, O, targets) in enumerate(val_loaders[mode]):
            inputs, O, targets = variable(inputs, volatile=True), variable(O), variable(targets)
            outputs = model(inputs, O)
            loss = criterion(outputs, targets)

            losses.append(loss.data[0])

            accuracy_scores += list(targets.data.cpu().numpy() == np.argmax(outputs.data.cpu().numpy(), axis=1))

        valid_loss[mode], valid_accuracy[mode] = np.mean(losses), np.mean(accuracy_scores)
        logging.info('{} | valid_loss: {}, valid_acc: {}'.format(mode, valid_loss[mode], valid_accuracy[mode]))
    weighted_loss = 0.7 * valid_loss['unalt'] + 0.3 * valid_loss['manip']
    weighted_acc = 0.7 * valid_accuracy['unalt'] + 0.3 * valid_accuracy['manip']
    logging.info('weighted loss: {} | weighted acc: {}\n'.format(weighted_loss, weighted_acc))
    return weighted_loss


def save_checkpoint(state, filename, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def get_validation_preds(val_loaders, model):
    dfs = dict()
    for mode in ('unalt', 'manip'):
        preds = None
        labels = []

        model.eval()

        for i, (inputs, O, targets) in tqdm(enumerate(val_loaders[mode])):
            inputs, O, targets = variable(inputs, volatile=True), variable(O), variable(targets)
            outputs = model(inputs, O)
            # print(list(targets))
            labels += list(targets.data.cpu().numpy().flatten())
            prediction = torch.nn.functional.softmax(outputs).data.cpu().numpy()
            preds = np.vstack([preds, prediction]) \
                if preds is not None else prediction
        labels = np.array(labels)
        print(preds.shape, labels.shape)
        
        df_data = np.append(preds, np.array(labels, copy=False, subok=True, ndmin=2).T, axis=1)
        dfs[mode] = pd.DataFrame(data=df_data, columns=utils.CLASSES + ['labels'])

    return dfs
