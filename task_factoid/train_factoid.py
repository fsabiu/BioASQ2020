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
tokenizer = BertTokenizer.from_pretrained("./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained("./transformers_models/biobert_factoid_pytorch", from_pt=True)
max_len = 512#TODO:gestire domanda con più token di max_len
epochs = 1
test_execution=True

###########################################################
# Pretrained model
model = model_creation(max_len, encoder)

# Data
x_data, y_data, answer_list = encode_daset(dataset_path, max_len, tokenizer, test_execution=test_execution)
x_data_test=x_data#TODO:aggiungere splittaggio test

# Training
trained_model=run_factoid_training(model, x_data, y_data, epochs)

#Evaluate
test_factoid_model(trained_model=model, tokenizer=tokenizer, x_data_test=x_data_test, answer_list=answer_list)
