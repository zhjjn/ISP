#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Data util for the model training and testing. Data packing and processing.

    Note:
    1. Training and testing data are different in that training requires some online data processing. Hence,
       they have their own separate data handler.
    2. For details on data online processing during training, see collate_fn of the DataHandler.
"""

import torch
from torch.utils import data as torch_data
from transformers.modeling_bart import shift_tokens_right
from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer
from src.utils.file_util import *
from src.utils.tensor_util import padding_list_of_tensors
from config import Config
from nltk.corpus import wordnet
import random


# Data handler for training and validation data
class Dataset(torch_data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, xs):
        super(Dataset, self).__init__()
        self.xs = xs
        self.num_total_seqs = len(self.xs)

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        return self.xs[index]


class DataHandler(object):
    """
    Data handler for modeling training. Handles training set and validation set.
    """

    def __init__(self):
        super(DataHandler, self).__init__()
        self.config = Config()
        self.sent_embed_mdl = SentenceTransformer(self.config.SENTENCE_EMBED_MODEL_NAME, device=self.config.PRE_DEVICE)
        pos_tags = load_json_file(self.config.PATH_TO_EXTRA_VOCAB)
        self.tokenizer = BartTokenizer.from_pretrained(self.config.PRETRAINED_CONDGEN_MODEL_NAME)
        num_added_toks = self.tokenizer.add_tokens(['pos_'+k for k in pos_tags.keys()])
        print(f'Added {num_added_toks} POS tokens for data loader!')

        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        if self.config.MODE == 'train':
            # load dictionaries
            self.type_format = {
                'v': 'verb',
                'a': 'adjective',
                'r': 'adverb',
                's': 'adjective'
            }
            self.raw_data = load_json_file(path_to_data_files['path_to_train_data'])
            self.definitions = {
                'wordnet': load_json_file(path_to_data_files['path_to_word_defn_wordnet']),
                'google': load_json_file(path_to_data_files['path_to_word_defn_google']),
                'wiktionary': load_json_file(path_to_data_files['path_to_word_defn_wiktionary']),
            }
            print('Loaded data from : {}'.format(path_to_data_files['path_to_train_data']))
        else:
            self.raw_data = load_json_file(path_to_data_files['path_to_test_data'])

    def init_generators(self):
        if self.config.MODE == 'train':

            self.train_dataset = Dataset(self.raw_data['train'])
            self.trainset_generator = torch_data.DataLoader(self.train_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=True,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=True)
            # data loader for validset
            self.valid_dataset = Dataset(self.raw_data['valid'][:3000])
            self.validset_generator = torch_data.DataLoader(self.valid_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=False,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=False)
        else:
            self.test_dataset = Dataset(self.raw_data)
            self.testset_generator = torch_data.DataLoader(self.test_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=False,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=False)

    def update_config(self):
        def get_batch_size(dataset_size):
            if dataset_size % self.config.BATCH_SIZE == 0:
                return dataset_size // self.config.BATCH_SIZE
            else:
                return dataset_size // self.config.BATCH_SIZE + 1

        # training parameters
        if self.config.MODE == 'train':
            self.config.train_size = len(self.train_dataset)
            self.config.valid_size = len(self.valid_dataset)
            print('Training dataset size: {}'.format(self.config.train_size))
            print('Validation dataset size: {}'.format(self.config.valid_size))
            self.config.num_batch_train = get_batch_size(self.config.train_size)
            self.config.num_batch_valid = get_batch_size(self.config.valid_size)
            self.config.NUM_TRAIN_STEPS = self.config.num_batch_train * self.config.BATCH_SIZE
        else:
            self.config.test_size = len(self.test_dataset)
            print('Testing dataset size: {}'.format(self.config.test_size))
            self.config.num_batch_test = get_batch_size(self.config.test_size)

    # Function to gather meanings from a word tuple
    @staticmethod
    def obtain_word_definitions(word):
        synsets = wordnet.synsets(word[0], pos=word[1])
        defns = [s.definition() for s in synsets]
        return defns

    # Function to generate word definition embeddings from a list of word definitions
    def obtain_word_definition_embeddings(self, word_definitions):
        word_definition_embeddings = []
        for defn in word_definitions:
            masked_word_defn_embed = self.sent_embed_mdl.encode(defn)
            word_definition_embeddings.append(masked_word_defn_embed.tolist())
        return word_definition_embeddings

    def generate_model_inputs(self, masked_sent, original_sent):
        input_encodings = self.tokenizer.batch_encode_plus(masked_sent, padding=True,
                                                      max_length=128, truncation=True)
        target_encodings = self.tokenizer.batch_encode_plus(original_sent, padding=True,
                                                       max_length=128, truncation=True)
        labels = target_encodings['input_ids']
        labels = torch.Tensor(labels).long()

        decoder_input_ids = shift_tokens_right(labels, self.config.PAD_IDX)
        labels[labels[:, :] == self.config.PAD_IDX] = -100

        return input_encodings['input_ids'], input_encodings['attention_mask'], decoder_input_ids.tolist(), labels.tolist()

    def select_word_definition(self, masked_word):
        source = ['google', 'wordnet', 'wiktionary']
        source = random.choice(source)
        masked_word, type = masked_word.split(':')
        type = self.type_format[type]
        if type not in self.definitions[source][masked_word]:
            source = 'wordnet'
        defn = self.definitions[source][masked_word][type]
        if len(defn) > self.config.MAX_NUM_DEFNS:
            defn = defn[:self.config.MAX_NUM_DEFNS]
        if source == 'wordnet':
            defn = [d + '.' for d in defn]
        # randomize the order of the defintions
        random.shuffle(defn)
        return defn

    def collate_fn(self, data):
        # 0. unpack data
        original_sent, masked_sent, masked_sent_lemmatized, masked_sent_ns, masked_sent_ns_lemmatized, masked_word, masked_pos = zip(*data)
        original_sent, masked_sent, masked_sent_lemmatized, masked_sent_ns, masked_sent_ns_lemmatized, masked_word, masked_pos = list(original_sent), \
                                                                                                                                 list(masked_sent), \
                                                                                                                                 list(masked_sent_lemmatized), \
                                                                                                                                 list(masked_sent_ns), \
                                                                                                                                 list(masked_sent_ns_lemmatized), \
                                                                                                                                 list(masked_word), \
                                                                                                                                 list(masked_pos)
        # Data pre-processing
        # 1. process the word definition and its masked embedding
        # select word definitions from dictionaries
        masked_word_defn = [self.select_word_definition(w) for w in masked_word]
        word_defn_embeds = self.obtain_word_definition_embeddings([defn for defn in masked_word_defn])
        word_defn_embeds = [torch.FloatTensor(seq) for seq in word_defn_embeds]
        word_defn_embeds, num_word_defns = padding_list_of_tensors(word_defn_embeds)
        word_defn_embeds = torch.vstack(word_defn_embeds)

        # 2. process input encodings
        if random.random() < self.config.no_stopwords_rate:
            # no stop words
            if random.random() < self.config.lemmatize_rate:
                masked_sent = masked_sent_ns_lemmatized
            else:
                masked_sent = masked_sent_ns
        else:
            # with stop words
            if random.random() < self.config.lemmatize_rate:
                masked_sent = masked_sent_lemmatized
            # else:
            #     masked_sent = masked_sent
        masked_sent = [' </s> '.join([s, 'pos_' + masked_pos[s_idx]]) for s_idx, s in enumerate(masked_sent)]

        input_ids, attention_mask, decoder_input_ids, labels = self.generate_model_inputs(masked_sent, original_sent)

        # 3. convert lists to tensors
        input_ids = [torch.Tensor(seq) for seq in input_ids]
        input_ids = torch.vstack(input_ids)
        attention_mask = [torch.Tensor(seq) for seq in attention_mask]
        attention_mask = torch.vstack(attention_mask)
        decoder_input_ids = [torch.Tensor(seq) for seq in decoder_input_ids]
        decoder_input_ids = torch.vstack(decoder_input_ids)
        labels = [torch.Tensor(seq) for seq in labels]
        labels = torch.vstack(labels)

        return {'input_ids': input_ids.long().to(self.config.DEVICE),
                'attention_mask': attention_mask.long().to(self.config.DEVICE),
                'decoder_input_ids': decoder_input_ids.long().to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE),
                'word_defn_embeds': word_defn_embeds.to(self.config.DEVICE),
                'num_word_defns': num_word_defns.long().to(self.config.DEVICE)}


class TestDataHandler(object):
    """
    Data Handler for model testing. Handles test set only.
    """
    def __init__(self):
        super(TestDataHandler, self).__init__()
        self.config = Config()
        self.sent_embed_mdl = SentenceTransformer(self.config.SENTENCE_EMBED_MODEL_NAME)# , device=self.config.PRE_DEVICE)
        self.tokenizer = BartTokenizer.from_pretrained(self.config.PRETRAINED_CONDGEN_MODEL_NAME)

        pos_tags = load_json_file(self.config.PATH_TO_EXTRA_VOCAB)
        self.tokenizer = BartTokenizer.from_pretrained(self.config.PRETRAINED_CONDGEN_MODEL_NAME)
        num_added_toks = self.tokenizer.add_tokens(['pos_'+k for k in pos_tags.keys()])
        print(f'Added {num_added_toks} POS tokens for data loader!')

        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        self.raw_data = load_json_file(path_to_data_files['path_to_test_data'])['test']

    def init_generators(self):

        self.test_dataset = Dataset(self.raw_data)
        self.testset_generator = torch_data.DataLoader(self.test_dataset,
                                                        batch_size=self.config.BATCH_SIZE,
                                                        collate_fn=self.collate_fn,
                                                        shuffle=False,
                                                        num_workers=self.config.NUM_WORKER,
                                                        drop_last=False)

    def update_config(self):
        def get_batch_size(dataset_size):
            if dataset_size % self.config.BATCH_SIZE == 0:
                return dataset_size // self.config.BATCH_SIZE
            else:
                return dataset_size // self.config.BATCH_SIZE + 1

        self.config.test_size = len(self.test_dataset)
        print('Testing dataset size: {}'.format(self.config.test_size))
        self.config.num_batch_test = get_batch_size(self.config.test_size)
        self.config.NUM_TRAIN_STEPS = 100000  # placeholder, no actual usage during test time.

    # Function to gather meanings from a word tuple
    @staticmethod
    def obtain_word_definitions(word):
        synsets = wordnet.synsets(word[0], pos=word[1])
        defns = [s.definition() for s in synsets]
        return defns

    # Function to generate word definition embeddings from a list of word definitions
    def obtain_word_definition_embeddings(self, word_definitions):
        word_definition_embeddings = []
        for defn in word_definitions:
            masked_word_defn_embed = self.sent_embed_mdl.encode(defn)
            word_definition_embeddings.append(masked_word_defn_embed.tolist())
        return word_definition_embeddings

    def generate_model_inputs(self, masked_sent, original_sent):
        input_encodings = self.tokenizer.batch_encode_plus(masked_sent, pad_to_max_length=True,
                                                      max_length=128, truncation=True)
        target_encodings = self.tokenizer.batch_encode_plus(original_sent, pad_to_max_length=True,
                                                       max_length=128, truncation=True)
        labels = target_encodings['input_ids']
        labels = torch.Tensor(labels).long()

        decoder_input_ids = shift_tokens_right(labels, self.config.PAD_IDX)
        labels[labels[:, :] == self.config.PAD_IDX] = -100

        return input_encodings['input_ids'], input_encodings['attention_mask'], decoder_input_ids.tolist(), labels.tolist()

    def collate_fn(self, data):
        # 1. unpack data
        masked_sent, masked_pos, masked_word_defn, original_sent = zip(*data)
        masked_sent, masked_pos, masked_word_defn, original_sent = list(masked_sent), list(masked_pos), list(masked_word_defn), list(original_sent)
        word_defn_embeds = self.obtain_word_definition_embeddings([[defn] for defn in masked_word_defn])
        # Data pre-processing
        # 1. process the masked embedding
        word_defn_embeds = [torch.FloatTensor(seq) for seq in word_defn_embeds]
        word_defn_embeds, num_word_defns = padding_list_of_tensors(word_defn_embeds)
        word_defn_embeds = torch.vstack(word_defn_embeds)
        masked_sent = [' </s> '.join([s, 'pos_' + masked_pos[s_idx]]) for s_idx, s in enumerate(masked_sent)]

        # 2. process input encodings
        input_ids, attention_mask, decoder_input_ids, labels = self.generate_model_inputs(masked_sent, original_sent)

        # 3. convert lists to tensors
        input_ids = [torch.Tensor(seq) for seq in input_ids]
        input_ids = torch.vstack(input_ids)
        attention_mask = [torch.Tensor(seq) for seq in attention_mask]
        attention_mask = torch.vstack(attention_mask)
        decoder_input_ids = [torch.Tensor(seq) for seq in decoder_input_ids]
        decoder_input_ids = torch.vstack(decoder_input_ids)
        labels = [torch.Tensor(seq) for seq in labels]
        labels = torch.vstack(labels)

        return {'input_ids': input_ids.long().to(self.config.DEVICE),
                'original_sent': original_sent,
                'attention_mask': attention_mask.long().to(self.config.DEVICE),
                'decoder_input_ids': decoder_input_ids.long().to(self.config.DEVICE),
                'labels': labels.long().to(self.config.DEVICE),
                'word_defn_embeds': word_defn_embeds.to(self.config.DEVICE),
                'num_word_defns': num_word_defns.long().to(self.config.DEVICE)}

