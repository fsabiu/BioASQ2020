import os
import re
import sys
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertModel
from functions_factoid import *

#######################################################
# Parameters
dataset_path = "./data/training8b.json"
tokenizer = BertTokenizer.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch", from_pt=True)
max_len = 512
batch_size = 64
epochs = 1
test_execution = True

###########################################################
# Pretrained model
model = model_creation(max_len, encoder)

# Data
x_data, y_data, answer_list = encode_daset(
    dataset_path, max_len, tokenizer, test_execution=5)
x_data_test = x_data  # TODO:aggiungere splittaggio test
x_data_val = x_data  # TODO:aggiungere splittaggio validation
y_data_val = y_data

# Training
trained_model = run_factoid_training(
    model, x_data, y_data, x_data_val, y_data_val, epochs, batch_size)

# Evaluate
test_factoid_model(trained_model=trained_model, tokenizer=tokenizer,
                   x_data_test=x_data_test, answer_list=answer_list)