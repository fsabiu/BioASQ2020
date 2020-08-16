import os
import re
import sys
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertModel
from tensorboard.plugins.hparams import api as hp
from functions_factoid import *

def execute_factoid(date,logdir,dataset_path,tokenizer,encoder,max_len,batch_size,epochs,learning_rate,test_execution=-1):

    # Pretrained model
    model = model_creation(max_len, learning_rate, encoder)

    # Data
    x_data, y_data, answer_list = encode_dataset(
        dataset_path, max_len, tokenizer, test_execution)
    x_data_test = x_data  # TODO:aggiungere splittaggio test

    # Training
    trained_model = run_factoid_training(
        model, x_data, y_data, epochs, batch_size,logdir)

    # Evaluate
    evaluation=test_factoid_model(trained_model=trained_model, tokenizer=tokenizer,
                    x_data_test=x_data_test, answer_list=answer_list)

    #Setup tensorboard

    HP_MODEL = hp.HParam('model')
    HP_EPOCHS = hp.HParam('n_epochs')
    HP_MAX_LEN = hp.HParam('max_len')
    HP_BATCH_SIZE = hp.HParam('batch_size')
    HP_LEARNING_RATE = hp.HParam('learning_rate')
    HP_DATE= hp.HParam('date')

    METRIC_STRICT_ACCURACY = 'strict_accuracy'
    METRIC_LENIENT_ACCURACY = 'lenient_accuracy'
    METRIC_MEAN_RECIPROCAL_RANK= 'mean_reciprocal_rank'

    hparams = {
            HP_MODEL :"biobert",
            HP_EPOCHS:epochs,
            HP_MAX_LEN:max_len,
            HP_BATCH_SIZE:batch_size,
            HP_LEARNING_RATE:learning_rate,
            HP_DATE : date
        }

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_STRICT_ACCURACY, float(evaluation["strict_accuracy"]), step=1)
        tf.summary.scalar(METRIC_LENIENT_ACCURACY, float(evaluation["lenient_accuracy"]), step=1)
        tf.summary.scalar(METRIC_MEAN_RECIPROCAL_RANK, float(evaluation["mean_reciprocal_rank"]), step=1)

#######################################################
# Parameters
date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
logdir = "./task_factoid/runs/"+str(date)

dataset_path = "./data/training8b.json"
tokenizer = BertTokenizer.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch")
encoder = TFBertModel.from_pretrained(
    "./transformers_models/biobert_factoid_pytorch", from_pt=True)

max_len = 512
batch_size = 10
epochs = 2
learning_rate = 0.00001
test_execution = 7


###########################################################

#execute_factoid(date,logdir,dataset_path,tokenizer,encoder,max_len,batch_size,epochs,learning_rate,test_execution)