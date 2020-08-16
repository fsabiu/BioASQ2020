import tensorflow as tf
from tensorflow import keras
from functions_factoid import *


dataset_path = "./data/training8b.json"
tokenizer = BertTokenizer.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch", from_pt=True)

max_len = 512
test_execution = 7
learning_rate=0.001

# Create a new model instance
model = model_creation(max_len, learning_rate, encoder)

# Restore the weights
model.load_weights('C:/Users/mgabr/Desktop/HLT/project/BioASQ2020/task_factoid/runs/20200816_170152/model.h5')
# Data
x_data, y_data, answer_list = encode_dataset(
    dataset_path, max_len, tokenizer, test_execution)
x_data_test = x_data  # TODO:aggiungere splittaggio test


# Evaluate
evaluation=test_factoid_model(trained_model=model, tokenizer=tokenizer,
                x_data_test=x_data_test, answer_list=answer_list)
