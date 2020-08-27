import tensorflow as tf
from tensorflow import keras
from functions_factoid import *


dataset_path_test = "./data/test_8b.json"
tokenizer = BertTokenizer.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch", from_pt=True)

max_len = 200
test_execution = -1
learning_rate = 5e-7

# Create a new model instance
model = model_creation(max_len, learning_rate, encoder)

# Restore the weights
model.load_weights(
    'C:/Users/mgabr/Desktop/HLT/project/BioASQ2020/task_factoid/model.h5')
# Data
x_data_test, _, answer_list_test = encode_dataset(
    dataset_path_test, max_len, tokenizer, test_execution)


# Evaluate
evaluation = test_factoid_model(trained_model=model, tokenizer=tokenizer,
                                x_data_test=x_data_test, answer_list=answer_list_test)
