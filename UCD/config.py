#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Experiment Configuration.
"""
from os.path import join, abspath, dirname
from src.utils.file_util import check_make_directory
import torch




class Config:
    ROOT = abspath(dirname(__file__))

    # Settings - (regularly changed)
    # ==============================================================================================================
    MODE = 'train'  # 'train' or 'test'
    MODEL_TYPE = 'bart'
    DATA_NAME = 'sample'
    MODEL_NAME = 'BART-UCD_{}_{}'.format(MODEL_TYPE, DATA_NAME)  # name the current training or testing model
    PATH_TO_META_DATA = './meta_data/meta_data_{}_{}.json'.format(MODEL_TYPE, DATA_NAME)
    PATH_TO_EXTRA_VOCAB = './data/pos_vocab/sample_POS_vocab.json'
    SENTENCE_EMBED_MODEL_NAME = 'roberta-base-nli-stsb-mean-tokens'
    # Book-keeping
    USE_GPU = True
    CONTINUE_TRAIN = False
    USE_TENSORBOARD = True
    VERBOSE = False  # display sampled prediction results
    NUM_WORKER = 0  # For multi-processing data set;

    # Checkpoint management
    PATH_TO_CHECKPOINT = join(ROOT, 'checkpoints/{}_{}.mdl')
    LOAD_CHECKPOINT_TYPE = 'latest'  # 'latest' or 'best
    check_make_directory('./checkpoints')

    # ++++++++++++++++++++++++++++++++++++++++++ PARAMETERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Train Parameters
    # ==============================================================================================================
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    VALID_FREQ = 1  # number of epochs to run validation
    SAVE_FREQ = 400  # number of steps to save train performance
    LOG_FREQ = 10
    DISPLAY_FREQ = 10  # number of steps to display train performance (only matters if VERBOSE==TRUE)
    LEARNING_RATE = 1e-5
    NUM_WARMUP_STEPS = 20000
    lemmatize_rate = 0.4
    no_stopwords_rate = 0.8
    MAX_NUM_DEFNS = 3  # maximize number of definitions per word allowed

    # Inference Parameters
    # ==============================================================================================================
    PATH_TO_SAVE_INFERENCE_RESULTS = join(ROOT, 'res/{}_inference_results.json'.format(MODEL_NAME))
    check_make_directory('./res')
    # Beam search parameters
    num_beams = 5
    min_length = 5
    max_length = 128
    do_sample = True
    num_return_sequences = 2
    use_cache = False
    is_encoder_decoder = True
    length_penalty = 0.8
    early_stopping = False
    repetition_penalty = 1
    no_repeat_ngram_size = 0
    output_scores = False
    return_dict_in_generate = False
    top_k = 100
    top_p = 0.50
    temperature = 1
    diversity_penalty = 0.2
    num_beam_groups = 1
    bad_words_ids = None

    # Data Parameters
    # ==============================================================================================================
    PRETRAINED_CONDGEN_MODEL_NAME = 'facebook/bart-large'
    PRETRAINED_CONDGEN_EMBED_DIM = 1024
    PRETRAINED_SENT_EMBED_DIM = 768
    MAX_SEQ_LEN = 128
    # Special symbols and indices
    PAD_IDX = pad_token_id = 1
    START_IDX = bos_token_id = decoder_start_token_id = 0
    END_IDX = eos_token_id = 2
    MASK_IDX = 50264
    UNK_IDX = 3

    # Model Parameters
    # ==============================================================================================================
    PRE_DEVICE = DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

    # Embedding Fusion Network
    HIGHWAY_NUM_LAYERS = 2

    # Mode based parameters
    if MODE != 'train':
        CONTINUE_TRAIN = True
        USE_TENSORBOARD = False
        BATCH_SIZE = 1
        LOAD_CHECKPOINT_TYPE = 'latest'



