import os
import re
import sys
import tensorflow as tf
import time
from tensorflow import keras
from transformers import BertTokenizer, TFBertModel
from tensorboard.plugins.hparams import api as hp
from functions_list import *

def execute_list(date, logdir, dataset_path, tokenizer, encoder, max_len, batch_size, epochs, learning_rate, test_execution=-1, save=False, evaluation=False):

    # Pretrained model
    model = model_creation(max_len, learning_rate, encoder)

    # Data
    x_data, y_data, answer_list = encode_dataset(
        dataset_path, max_len, tokenizer, test_execution)

    # TODO:aggiungere splittaggio test
    #x_data_test = x_data
    x_data_test, y_data_test, answer_list_test = encode_dataset(
        dataset_path, max_len, tokenizer, 10)

    #######
    # Training
    #trained_model = run_factoid_training(
    #    model, x_data, y_data, epochs, batch_size, logdir)

    # Evaluate
    results_evaluation = {"mean_precison": 0.0,
                          "mean_recall": 0.0, "mean_f1": 0.0}

    if(evaluation == True):
        results_evaluation = test_list_model(trained_model=model, tokenizer=tokenizer,
                                                x_data_test=x_data_test, answer_list=answer_list_test)

    if(save == True):
        trained_model.save_weights(logdir+"model.h5", save_format='h5')

    # Setup tensorboard

    HP_MODEL = hp.HParam('model')
    HP_EPOCHS = hp.HParam('n_epochs')
    HP_MAX_LEN = hp.HParam('max_len')
    HP_BATCH_SIZE = hp.HParam('batch_size')
    HP_LEARNING_RATE = hp.HParam('learning_rate')
    HP_DATE = hp.HParam('date')

    METRIC_MEAN_PRECISION = 'mean_precison'
    METRIC_MEAN_RECALL = 'mean_recall'
    METRIC_MEAN_F1 = 'mean_f1'

    hparams = {
        HP_MODEL: "biobert",
        HP_EPOCHS: epochs,
        HP_MAX_LEN: max_len,
        HP_BATCH_SIZE: batch_size,
        HP_LEARNING_RATE: learning_rate,
        HP_DATE: date
    }

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_MEAN_PRECISION, float(
            results_evaluation["mean_precison"]), step=1)
        tf.summary.scalar(METRIC_MEAN_RECALL, float(
            results_evaluation["mean_recall"]), step=1)
        tf.summary.scalar(METRIC_MEAN_F1, float(
            results_evaluation["mean_f1"]), step=1)


#######################################################
# Parameters
date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
logdir = "./task_list/runs/"+str(date)

dataset_path = "./data/training8b.json"
tokenizer = BertTokenizer.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch", from_pt=True)

max_len = 512
batch_size = 5
epochs = 1
learning_rate = 0.00001
test_execution = 7


###########################################################
if __name__ == "__main__":
    start_time = time.time()
    execute_list(date, logdir, dataset_path, tokenizer, encoder,
                    max_len, batch_size, epochs, learning_rate, test_execution=test_execution, evaluation=True)
    print("--- %s seconds ---" % (time.time() - start_time))
