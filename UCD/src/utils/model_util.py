"""
Helper functions for checkpoint management; saving and loading model checkpoints and other aux stuff.
"""
import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


# Manage model and checkpoint loading
def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar', device='cpu'):
    '''
    Load a checkpoint.

    # Parameters

    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :param filename: (str) path to saved checkpoint

    # Returns

    :return model: loaded model
    :return optimizer: optimizer with right parameters
    :return start_epoch: (int) epoch number load from model
    '''

    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        if device == 'cpu':
            checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
        else:
            checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])  # reload
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch


def load_init_model(model_module, config):
    """
    Initialize a model and load a checkpoint if so desired (if the checkpoint is available.)

    # Parameters

    :param model_module: the class of the model.
    :param config: config class that contains all the parameters

    # Returns

    :return model: initialized model (loaded checkpoint)
    :return optimizer: initialized optimizer
    :return epoch_start: the starting epoch to continue the training
    """

    epoch_start = 0

    model = model_module(config).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.NUM_WARMUP_STEPS,
                                                config.NUM_TRAIN_STEPS)
    save_path = config.PATH_TO_CHECKPOINT.format(config.MODEL_NAME, config.LOAD_CHECKPOINT_TYPE)

    if os.path.exists(save_path) and config.CONTINUE_TRAIN:
        print('Loading model from {}'.format(save_path))
        model, optimizer, scheduler, epoch_start = load_checkpoint(model, optimizer, scheduler, save_path, config.DEVICE)
    else:
        print('=> No checkpoint found! Train from scratch!')
    model.eval()

    return model, optimizer, scheduler, epoch_start


def save_model(save_path, model, optimizer, scheduler, epoch):
    """
    Save the model, loss, and optimizer to a checkpoint
    :param save_path: path to save the checkpoint
    :param model: the model to be saved
    :param optimizer: the optimizer used during training.
    :param scheduler: the scheduler used during training.
    :param epoch: the current number of epoch.
    :return:
    """
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
    torch.save(state, save_path)


def count_parameters(model):
    """
    Count the total number of parameters in the model.
    :param model: model to be counted.
    :return: int: number of parameters
    """
    # get total size of *trainable* parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

