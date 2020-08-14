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
batch_size = 10
epochs = 20
learning_rate=0.00001
test_execution = 7

###########################################################
# Pretrained model
model = model_creation(max_len, learning_rate, encoder)

# Data
x_data, y_data, answer_list = encode_dataset(
    dataset_path, max_len, tokenizer, test_execution)
x_data_test = x_data  # TODO:aggiungere splittaggio test

# Training
#trained_model = run_factoid_training(
#    model, x_data, y_data, epochs, batch_size)

# Evaluate
test_factoid_model(trained_model=model, tokenizer=tokenizer,
                   x_data_test=x_data_test, answer_list=answer_list)
