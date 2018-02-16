import os
import logging
import numpy as np

import torch
import torch.nn as nn

from . import utils
from ..mymodels import densenet, dpn, resnext, se_resnet, seresnext
from .. import config

cuda_is_available = torch.cuda.is_available()


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


def train_and_validate(train_data_loader, valid_loaders, model, optimizer,
                       scheduler, loss_fn, experiment_name):
    checkpoint_dir = os.getcwd() + '/models/{}'.format(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tracker = Tracker()
    cyclic_lr_started = False
    valid_losses = []
    epoch = 1
    while True:
        train(train_data_loader, model,
              optimizer, loss_fn, epoch)
        val_loss = validate(valid_loaders, model, loss_fn)
        valid_losses.append(val_loss)

        if tracker.should_save(val_loss):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            },
                '{experiment_name}_{epoch}_{val_loss}.pth'.format(
                    experiment_name=experiment_name, epoch=epoch,
                    val_loss=val_loss),
                checkpoint_dir,
            )
        if tracker.not_updated_for() >= 8 and not cyclic_lr_started:
            cyclic_lr_started = True
            init_lr = config.INIT_CYCLIC_LR if 'dpn' not in experiment_name else 1e-4
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
            epoch += config.CYCLE_LEN - (epoch % config.CYCLE_LEN) - 1
            finish_at = epoch + 3 * config.CYCLE_LEN + 1
            logging.info('Starting cyclic lr with init_lr={}'.format(init_lr))
        if not cyclic_lr_started:
            scheduler.step(val_loss, epoch)
        else:
            for param_group in optimizer.param_groups:
                new_lr = cyclic_lr(epoch, init_lr=config.INIT_CYCLIC_LR,
                                   num_epochs_per_cycle=config.CYCLE_LEN)
                param_group['lr'] = new_lr
        epoch += 1
        if cyclic_lr_started and epoch == finish_at:
            break
    return model


def train(train_loader, model, optimizer, criterion, epoch):
    losses = []

    model.train()
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    logging.info('Epoch: {} | lrs: {}'.format(epoch, lrs))
    for i, (inputs, O, targets) in enumerate(train_loader):
        inputs, O, targets = variable(inputs), variable(O), variable(targets)
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


def make_model(model_name):
    if model_name.startswith('densenet201'):
        model = densenet.densenet201(num_classes=len(utils.CLASSES), pretrained=True)
    elif model_name == 'dpn92':
        model = dpn.dpn92(num_classes=len(utils.CLASSES), pretrained='imagenet+5k')
    elif model_name.startswith('resnext101'):
        model = resnext.resnext101_32x4d(num_classes=len(utils.CLASSES), pretrained='imagenet')
    elif model_name == 'densenet161':
        model = densenet.densenet161(num_classes=len(utils.CLASSES), pretrained=True)
    elif model_name == 'dpn98':
        model = dpn.dpn98(num_classes=len(utils.CLASSES), pretrained='imagenet')
    elif model_name == 'se_resnet50':  # FIXME Add imagenet pretrained weights
        model = se_resnet.se_resnet50(num_classes=len(utils.CLASSES)).cuda()
        return model
    elif model_name == 'se_resnext50':  # FIXME Add imagenet pretrained weights
        model = seresnext.se_resnext50(num_classes=len(utils.CLASSES), pretrained=True).cuda()
        return model
    else:
        raise RuntimeError('Unknown model name: {}'.format(model_name))
    model = nn.DataParallel(model).cuda()
    return model
