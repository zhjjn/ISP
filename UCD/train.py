#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for Seq2seq model.
"""
from datetime import datetime
from tqdm import trange
from tensorboardX import SummaryWriter
from src.utils.data_util import DataHandler
from src.utils.model_util import load_init_model, save_model
from src.train_valid_test_step import train_step, valid_step
from config import Config
from torch.multiprocessing import set_start_method
from src.model.masked_conditional_generation_model_huggingface import MaskedConditionalGenerationModelHF


try:
    set_start_method('spawn')
except RuntimeError:
    pass


def train():
    """
    The training script for the Relation2Vec model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandler()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize and load model
    save_path = Config.PATH_TO_CHECKPOINT
    model, optimizer, scheduler, epoch_start = load_init_model(MaskedConditionalGenerationModelHF, data_handler.config)

    # Freeze the BART encoder layers
    for param in model.bart.base_model.encoder.parameters():
        param.requires_grad = False

    # Book-keeping for fine-tuneing
    # Add Tensorboard writer
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs/{}_{}'.format(Config.MODEL_NAME, datetime.today().strftime('%Y-%m-%d')))
    # Book-keeping info
    best_valid_loss = float('inf')

    # Train model
    # ---------------------------------------------------------------------------------
    ebar = trange(epoch_start, Config.NUM_EPOCHS, desc='EPOCH', ncols=130, leave=True)
    for epoch in ebar:
        # Training
        _, _ = train_step(model, optimizer, scheduler, data_handler, epoch, writer)

        # Validation
        if epoch % Config.VALID_FREQ == 0:
            valid_loss, valid_acc = valid_step(model, data_handler)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                # save the best model seen so far
                save_model(save_path.format(Config.MODEL_NAME, 'best'), model, optimizer, scheduler, epoch)
            if Config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_acc', valid_acc, epoch)

        # save the latest model
        save_model(save_path.format(Config.MODEL_NAME, f'latest-{epoch}'), model, optimizer, scheduler, epoch)

    return


if __name__ == '__main__':
    train()


