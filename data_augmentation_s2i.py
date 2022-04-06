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
print("Task: S2I. Mode: " + args.mode + " Model: " + args.model)


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
        
S2I_model_args = T5Args()
S2I_model_args.do_sample = True
S2I_model_args.eval_batch_size = 32
S2I_model_args.evaluate_during_training = True
S2I_model_args.evaluate_during_training_steps = 25000
S2I_model_args.evaluate_during_training_verbose = True
S2I_model_args.fp16 = False
S2I_model_args.learning_rate = 5e-5
S2I_model_args.max_length = 128
S2I_model_args.max_seq_length = 128
S2I_model_args.num_beams = None
S2I_model_args.num_return_sequences = 3
S2I_model_args.num_train_epochs = 5
S2I_model_args.overwrite_output_dir = True
S2I_model_args.reprocess_input_data = True
S2I_model_args.save_eval_checkpoints = False
S2I_model_args.save_steps = -1
S2I_model_args.top_k = 50
S2I_model_args.top_p = 0.95
S2I_model_args.train_batch_size = 8
S2I_model_args.use_multiprocessing = False
S2I_model_args.output_dir = "S2I_" + args.model + "_model/outputs/"
S2I_model_args.best_model_dir = "S2I_" + args.model + "_model/outputs/best_model"
S2I_model_args.cache_dir = "S2I_" + args.model + "_model/cache_dir"

if args.mode == 'train':
    S2I_model = MODEL_CLASSES[args.model][0](encoder_decoder_type = args.model, encoder_decoder_name = MODEL_CLASSES[args.model][2], cuda_device = 1, args= S2I_model_args,)

    if args.model == 't5':
        S2I_model.train_model(P_s2i_df[["prefix", "input_text", "target_text"]], eval_data = P_s2i_df[["prefix", "input_text", "target_text"]])
    else:
        S2I_model.train_model(P_s2i_df[["input_text", "target_text"]], eval_data = P_s2i_df[["input_text", "target_text"]])

if args.mode == 'inference':
    I2S_M_data = pd.read_csv("./middle/M_I2S_" + str(r) + ".csv")

    Idiom_M = I2S_M_data["Idiom_M"].tolist()
    Simple_M = I2S_M_data["Simple_M"].tolist()
    idioms_M = I2S_M_data["idioms_M"].tolist()
    to_predict = Simple_M
    if args.model == 't5':
        to_predict = ["s2i: " + t for t in to_predict]

    S2I_model = MODEL_CLASSES[args.model][0](encoder_decoder_type = args.model, encoder_decoder_name = S2I_model_args.best_model_dir, cuda_device = 1, args= S2I_model_args,)

    preds = S2I_model.predict(to_predict)
    
    preds_I = [p[0] for p in preds]
    I_M = []
    S_M = []
    idm_M = []
    for i in range(len(to_predict)):
      if to_predict[i].replace("s2i: ", "") != preds_I[i] and Idiom_M[i] == preds_I[i]:
        I_M.append(Idiom_M[i])
        S_M.append(to_predict[i].replace("s2i: ", ""))
        idm_M.append(idioms_M[i])
        
    I_M_num = len(I_M)
    print("Added new examples: ", I_M_num)
    
    headers = ['Idiom', 'Idiomatic_Sent', 'Literal_Sent', 'Idiomatic_Sent_wIdiom']
    output_row = []
    for i in range(len(I_M)):
      output_row.append([idm_M[i], I_M[i], S_M[i], I_M[i] + " </s> " + idm_M[i]])

    data_P = dataset_P[["Idiom", "Idiomatic_Sent", "Literal_Sent", "Idiomatic_Sent_wIdiom"]]
    I_P = data_P['Idiomatic_Sent'].tolist()
    S_P = data_P['Literal_Sent'].tolist()
    I_P_wIdiom = data_P['Idiomatic_Sent_wIdiom'].tolist()
    idm_P = data_P['Idiom'].tolist()

    for i in range(len(I_P)):
      output_row.append([idm_P[i], I_P[i], S_P[i], I_P_wIdiom[i]])

    sentence = M_df['reference'].tolist()
    idiom = M_df['target_text'].tolist()
    M_sentence = []
    M_idiom = []
    M_row = []

    M_headers = ['Idiom', 'Sentence', 'Sentence_wIdiom']
    for i in range(len(sentence)):
      if sentence[i] not in I_M:
        M_sentence.append(sentence[i])
        M_idiom.append(idiom[i])
        M_row.append([idiom[i], sentence[i], sentence[i] + " </s> " + idiom[i]])
    
    r += 1

    with open("./data/data_P_wIdiom_" + str(r) + ".csv",'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(output_row)

    with open("./data/data_M_wIdiom_" + str(r) + ".csv",'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(M_headers)
        f_csv.writerows(M_row)