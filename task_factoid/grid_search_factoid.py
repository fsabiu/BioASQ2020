from itertools import product
from datetime import datetime
from transformers import BertTokenizer, TFBertModel
from train_factoid import execute_factoid

grid_params = {'max_len': [512],
               'batch_size': [10],
               'epochs': [1, 2],
               'learning_rate': [0.00001, 0.0001]}

keys, values = zip(*grid_params.items())
params_list = [dict(zip(keys, v)) for v in product(*values)]

print("Run grid search with ",len(params_list)," models")

for elem in params_list:
    date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
    logdir = "./task_factoid/runs/"+str(date)

    dataset_path_train = "./data/train_8b.json"
    dataset_path_test = "./data/test_8b.json"
    tokenizer = BertTokenizer.from_pretrained(
        "./transformers_models/biobert_factoid_pytorch")
    encoder = TFBertModel.from_pretrained(
        "./transformers_models/biobert_factoid_pytorch", from_pt=True)

    max_len = elem["max_len"]
    batch_size = elem["batch_size"]
    epochs = elem["epochs"]
    learning_rate = elem["learning_rate"]

    # Delete for complete run
    test_execution = 7

    execute_factoid(date=date, logdir=logdir, dataset_path_train=dataset_path_train, dataset_path_test=dataset_path_test, tokenizer=tokenizer, encoder=encoder,
                    max_len=max_len, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, test_execution=test_execution)
