from functions_yesno import get_embedding, enconde_dataset, model_creation, run_yesno_training, evaluate_yes_no, test_final_model, evaluate_model, EPOCHS
import numpy as np
from random import seed
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

def execute_yesno(
        x_train, y_train,
        hidden_layers, hidden_units, act_funct, learning_rate, optimizer, pool_size, batch_size,
        x_test = None, y_test = None,
        date = None, logdir = None
    ):
    
    date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/" if date is None else date
    logdir = "./task_yesno/runs/"+str(date) if logdir is None else logdir

    model = model_creation(hidden_layers, hidden_units, act_funct, learning_rate, optimizer)
    training_model = run_yesno_training(model, x_train, y_train, pool_size=pool_size, batch_size=batch_size, logdir=logdir)

    results_evaluation = {'accuracy':0,'macro_average_f_measure':0}
    if x_test is not None and y_test is not None:
        results_evaluation = evaluate_model(training_model, x_test, y_test)

    # Setup tensorboard
    HP_MODEL = hp.HParam('model')
    HP_EPOCHS = hp.HParam('n_epochs')
    HP_HIDDEN_LAYERS = hp.HParam('hidden_layers')
    HP_HIDDEN_UNITS = hp.HParam('hidden_units')
    HP_ACT_FUNCT = hp.HParam('act_funct')
    HP_LEARNING_RATE = hp.HParam('learning_rate')
    HP_OPTIMIZER = hp.HParam('optimizer')
    HP_POOL_SIZE = hp.HParam('pool_size')
    HP_BATCH_SIZE = hp.HParam('batch_size')
    HP_DATE = hp.HParam('date')

    METRIC_ACCURACY = 'accuracy'
    METRIC_F_MEASURE = 'macro_average_f_measure'

    batch_size = 0 if batch_size is None else batch_size

    hparams = {
        HP_MODEL: "elmopubmed",
        HP_EPOCHS: EPOCHS,
        HP_HIDDEN_LAYERS: hidden_layers,
        HP_HIDDEN_UNITS: hidden_units,
        HP_ACT_FUNCT: act_funct,
        HP_OPTIMIZER: optimizer.__name__,
        HP_POOL_SIZE: pool_size,
        HP_BATCH_SIZE: batch_size,
        HP_LEARNING_RATE: learning_rate,
        HP_DATE: date
    }

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_ACCURACY, float(
            results_evaluation["accuracy"]), step=1)
        tf.summary.scalar(METRIC_F_MEASURE, float(
            results_evaluation["macro_average_f_measure"]), step=1)

    return training_model



if __name__ == "__main__":
    print("Getting embedding ") # To create embedding model see load_training_data notebook
    emb = get_embedding(file_embedding = "./embedding_yes_no_augmented.emb")
    print("Getting data")
    # x_train, y_train, x_test, y_test = enconde_dataset(emb, pool_size=1, test_size=0.2)
    x_train, y_train = enconde_dataset(emb)

    hidden_layers = 2
    hidden_units = 150
    act_funct = 'relu'
    learning_rate = 1e-5
    optimizer = RMSprop
    pool_size = 1
    batch_size = None

    # execute_yesno(x_train, y_train, hidden_layers, hidden_units, act_funct, learning_rate, optimizer, pool_size, batch_size, x_test, y_test)
    final_model = execute_yesno(x_train, y_train, hidden_layers, hidden_units, act_funct, learning_rate, optimizer, pool_size, batch_size)


