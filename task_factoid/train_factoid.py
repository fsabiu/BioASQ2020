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
max_len = 512
epochs = 10

###########################################################
# Pretrained model
model = model_creation(max_len, encoder)

# Data
x_data, y_data = encode_daset(dataset_path, max_len, tokenizer)

# Training
run_factoid_training(model, x_data, y_data, epochs)
