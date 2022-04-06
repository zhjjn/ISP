import warnings
import csv
import pandas as pd
import os
from datetime import datetime
import logging

from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.t5 import T5Model, T5Args
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

MODEL_CLASSES = {
    "t5": (T5Model, T5Args(), "t5-large"),
    "bart": (Seq2SeqModel, Seq2SeqArgs(), 'facebook/bart-large'),
}

def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

parser = argparse.ArgumentParser()
parser.add_argument('--round', action='store', dest='round',help='Specify the round')
parser.add_argument('--mode', action='store', dest='mode',help='Specify the mode')
parser.add_argument('--model', action='store', dest='model', help = 'Specify the model')
args = parser.parse_args()

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))

print("Task: I2S. Mode: " + args.mode + " Model: " + args.model)


r = int(args.round)
dataset_P = pd.read_csv("./data/data_P_wIdiom_" + str(r) + ".csv")
dataset_M = pd.read_csv("./data/data_M_wIdiom_" + str(r) + ".csv")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

P_df = dataset_P[["Idiomatic_Sent", "Literal_Sent", "Idiomatic_Sent_wIdiom"]]
M_df = dataset_M[["Sentence", "Idiom", "Sentence_wIdiom"]]

P_i2s_df = P_df.rename(
        columns={"Idiomatic_Sent_wIdiom": "input_text", "Literal_Sent": "target_text", "Idiomatic_Sent": "reference"}
)

P_s2i_df = P_df.rename(
        columns={"Idiomatic_Sent": "target_text", "Literal_Sent": "input_text", "Idiomatic_Sent_wIdiom": "reference"}
)

M_df = M_df.rename(
        columns={"Sentence_wIdiom": "input_text", 'Idiom': "target_text", "Sentence": "reference"}
)
    
P_i2s_df['prefix'] = "i2s"
P_s2i_df['prefix'] = "s2i"

P_df = P_df.dropna()
M_df = M_df.dropna()

P_i2s_df["input_text"] = P_i2s_df["input_text"].apply(clean_unnecessary_spaces)
P_i2s_df["target_text"] = P_i2s_df["target_text"].apply(clean_unnecessary_spaces)
P_i2s_df["reference"] = P_i2s_df["reference"].apply(clean_unnecessary_spaces)

P_s2i_df["input_text"] = P_s2i_df["input_text"].apply(clean_unnecessary_spaces)
P_s2i_df["target_text"] = P_s2i_df["target_text"].apply(clean_unnecessary_spaces)
P_s2i_df["reference"] = P_s2i_df["reference"].apply(clean_unnecessary_spaces)

M_df["input_text"] = M_df["input_text"].apply(clean_unnecessary_spaces)
M_df["reference"] = M_df["reference"].apply(clean_unnecessary_spaces)

#test_data["Literal_Part"] = test_data["Literal_Part"].apply(clean_unnecessary_spaces)

print(len(P_i2s_df['input_text'].tolist()))
print(len(M_df['input_text'].tolist()))
    
model_args = T5Args()
model_args.do_sample = True
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 25000
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_length = 128
model_args.max_seq_length = 128
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.num_train_epochs = 5
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.top_k = 50
model_args.top_p = 0.95
model_args.train_batch_size = 8
model_args.use_multiprocessing = False
model_args.output_dir = "I2S_" + args.model + "_model/outputs/"
model_args.best_model_dir = "I2S_" + args.model + "_model/outputs/best_model"
model_args.cache_dir = "I2S_" + args.model + "_model/cache_dir"
#model_args.wandb_project = "MWE with BART"

torch.cuda.empty_cache()

if args.mode == 'train':
    I2S_model = MODEL_CLASSES[args.model][0](encoder_decoder_type = args.model, encoder_decoder_name = MODEL_CLASSES[args.model][2], cuda_device = 0, args=model_args,)

    if args.model == 't5':
        I2S_model.train_model(P_i2s_df[["prefix", "input_text", "target_text"]], eval_data = P_i2s_df[["prefix", "input_text", "target_text"]])
    else:
        I2S_model.train_model(P_i2s_df[["input_text", "target_text"]], eval_data = P_i2s_df[["input_text", "target_text"]])

if args.mode == 'inference':
    to_predict = M_df["input_text"].tolist()
    if args.model == 't5':
        to_predict = ["i2s: " + t for t in to_predict]

    I2S_model = MODEL_CLASSES[args.model][0](encoder_decoder_type = args.model, encoder_decoder_name = model_args.best_model_dir, cuda_device = 0, args=model_args,)

    preds = I2S_model.predict(to_predict)
    
    Idiom_M = []
    idioms_M = []
    idioms = M_df["target_text"].tolist()
    Simple_M = []
    S_M = preds
    to_predict_ = M_df["reference"].tolist()
    for i in range(len(to_predict)):
      best = 1
      best_s = ""
      for pred in S_M[i]:
        diff = len(list(set(pred.split()) - set(to_predict_[i].split())))
        if diff > best:
          best = diff
          best_s = pred
      if best_s == "":
        continue

      Idiom_M.append(to_predict_[i])
      Simple_M.append(best_s)
      idioms_M.append(idioms[i])
    
    headers = ['idioms_M', 'Idiom_M', 'Simple_M']
    output_row = []
    for i in range(len(Idiom_M)):
      output_row.append([idioms_M[i], Idiom_M[i], Simple_M[i]])


    with open("./middle/M_I2S_" + str(r) + ".csv",'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(output_row)