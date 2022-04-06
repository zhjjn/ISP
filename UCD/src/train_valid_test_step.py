#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training/validation/inference step for BART-UCD model.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.eval_util import AverageMeter, compute_performance
from src.utils.model_util import save_model


def train_step(model, optimizer, scheduler, data_handler, epoch, writer):
    """Train model for one epoch"""
    model.train()
    # performance recorders
    loss_epoch = AverageMeter()
    acc_epoch = AverageMeter()

    # train data for a single epoch
    bbar = tqdm(enumerate(data_handler.trainset_generator), ncols=100, leave=False,
                total=data_handler.config.num_batch_train)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['input_ids'].shape[0]

        # model forward pass to compute the node embeddings
        outputs = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            word_defn_embed=data['word_defn_embeds'],
            num_word_defns=data['num_word_defns'],
            decoder_input_ids=data['decoder_input_ids'],
            labels=data['labels'],
            return_dict=False
        )
        loss = outputs['loss']

        # compute negative sampling loss and update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        ys_ = F.log_softmax(outputs['logits'], dim=-1)
        ys_ = torch.argmax(ys_, dim=-1).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        seq_acc = compute_performance(data['labels'].cpu().detach().numpy(), ys_, data_handler.config)
        acc_epoch.update(seq_acc, batch_size)

        # set display bar
        bbar.set_description("Phase: [Train] | Train Loss: {:.5f} | Acc: {:.3f} |<~.~>|".format(loss, seq_acc))
        if idx % data_handler.config.LOG_FREQ == 0:
            if data_handler.config.USE_TENSORBOARD:
                writer.add_scalar('train_loss', loss_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
                writer.add_scalar('train_acc', acc_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
        if idx % data_handler.config.SAVE_FREQ == 0:
            save_path = data_handler.config.PATH_TO_CHECKPOINT
            save_model(save_path.format(data_handler.config.MODEL_NAME, 'latest'), model, optimizer, scheduler, epoch)

    return loss_epoch.avg, acc_epoch.avg


def valid_step(model, data_handler):
    """Valid model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    acc_epoch = AverageMeter()

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.validset_generator), ncols=100, leave=False,
                total=data_handler.config.num_batch_valid)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['input_ids'].shape[0]

        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            outputs = model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                word_defn_embed=data['word_defn_embeds'],
                num_word_defns=data['num_word_defns'],
                decoder_input_ids=data['decoder_input_ids'],
                labels=data['labels'],
                return_dict=False
            )
            loss = outputs['loss']

        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        ys_ = F.log_softmax(outputs['logits'], dim=-1)
        ys_ = torch.argmax(ys_, dim=-1).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        seq_acc = compute_performance(data['labels'].cpu().detach().numpy(), ys_, data_handler.config)
        acc_epoch.update(seq_acc, batch_size)

        # random sample to show
        bbar.set_description("Phase: [Valid] | Valid Loss: {:.3f} | Acc: {:.3f} |".format(loss,
                                                                                          seq_acc))
    return loss_epoch.avg, acc_epoch.avg


