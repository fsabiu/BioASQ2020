from itertools import product
from datetime import datetime
from transformers import BertTokenizer, TFBertModel
from train_yesno import execute_yesno
from functions_yesno import get_embedding, enconde_dataset
import tensorflow as tf

if __name__ == "__main__":
    grid_params = {'hidden_layers': [1],
               'hidden_units': [5000],
               'act_funct': ['sigmoid'],
               'learning_rate': [1e-5]}

    keys, values = zip(*grid_params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    print("Run grid search with ",len(params_list)," models")

    print("Getting embedding ")
    emb = get_embedding(file_embedding = "./embedding_yes_no_augmented.emb")
    print("Getting data")
    x_train, y_train, x_test, y_test = enconde_dataset(emb)

    for elem in params_list:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
        logdir = "./task_factoid/runs/"+str(date)

        hidden_layers = elem['hidden_layers']
        hidden_units = elem['hidden_units']
        act_funct = elem['act_funct']
        learning_rate = elem['learning_rate']

        execute_yesno(x_train, y_train, hidden_layers, hidden_units, act_funct, learning_rate, x_test, y_test)