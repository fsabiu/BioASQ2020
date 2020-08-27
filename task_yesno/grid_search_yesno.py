from itertools import product
from datetime import datetime
from transformers import BertTokenizer, TFBertModel
from train_yesno import execute_yesno
from functions_yesno import get_embedding, enconde_dataset
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

if __name__ == "__main__":
    grid_params = {
                'hidden_layers': [1, 2],
                'hidden_units': [50, 100, 150],
                'act_funct': ['relu', 'tanh'],
                'learning_rate': [1e-6, 1e-5, 1e-7],
                'optimizer': [Adam, RMSprop],
                'pool_size': [1, 4, 8],
                'batch_size': [None, 8, 32]
            }

    keys, values = zip(*grid_params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    print("Run grid search with ",len(params_list)," models")

    print("Getting embedding ")
    emb = get_embedding(file_embedding = "./embedding_yes_no_augmented.emb")
    print("Getting data")
    x_train, y_train = enconde_dataset(emb)

    for elem in params_list:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
        logdir = "./task_yesno/runs/"+str(date)

        hidden_layers = elem['hidden_layers']
        hidden_units = elem['hidden_units']
        act_funct = elem['act_funct']
        learning_rate = elem['learning_rate']
        optimizer = elem['optimizer']
        pool_size = elem['pool_size']
        batch_size = elem['batch_size']

        execute_yesno(x_train, y_train, hidden_layers, hidden_units, act_funct, learning_rate, optimizer, pool_size, batch_size)