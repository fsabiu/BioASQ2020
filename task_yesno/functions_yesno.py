import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from flair.data import Sentence
import progressbar
import pickle
import sys, os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_data, generate_embeddings_yesno, generate_embeddings_yesno_pooling, load_embeddings

from flair.embeddings import ELMoEmbeddings, DocumentPoolEmbeddings
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, MaxPooling1D, Embedding, InputLayer
from tensorflow.keras.models import Model
import numpy as np

BATCH_SIZE = 4
VALIDATION_SPLIT = 0.15
EPOCHS = 50


def evaluate_yes_no(predicted, target):
    accuracy=accuracy_score(y_true=target, y_pred=predicted)

    f1_y=f1_score(y_true=target, y_pred=predicted, average='binary', pos_label=1)
    f1_n=f1_score(y_true=target, y_pred=predicted, average='binary', pos_label=0)
    macro_average_f_measure=(f1_y+f1_n)/2

    return {'accuracy':accuracy,'macro_average_f_measure':macro_average_f_measure}

def get_embedding(data = None, file_embedding = None):
    if file_embedding is None:
        embeddings_elmo_pubmed = ELMoEmbeddings('pubmed') 
        pooling_model = DocumentPoolEmbeddings([embeddings_elmo_pubmed])
        embeddings = generate_embeddings_yesno_pooling(pooling_model, data, "embedding_yes_no.emb")
    else:
        # ../data/embedding_yes_no.emb
        embeddings = load_embeddings(file_embedding)
    return embeddings


def enconde_dataset(embeddings, pool_size=1, test_size=None):
    # Data includes questions (0) and snippets (2)
    data = np.array([np.concatenate((np.array(el[0]), np.array(el[2])),axis=None) for el in embeddings])

    labels = np.array([el[1] for el in embeddings])

    # Lo split potrebbe essere inutile dal momento che il validation è creato dal fit del modello
    if test_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
        x_train = apply_pooling(x_train, pool_size)
        x_test = apply_pooling(x_test, pool_size)
        return x_train, y_train, x_test, y_test
    else: 
        return data, labels

'''
def hyperparameter_creation(hp_dict):
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete(hp_dict['num_units']))
    HP_NUM_HIDDENS = hp.HParam('num_hiddens', hp.Discrete(hp_dict['num_hiddens']))
    HP_ACT_FUN = hp.HParam('act_fun', hp.Discrete(hp_dict['act_fun']))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'

    directory_results = 'logs/hparam_tuning'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with tf.summary.create_file_writer(directory_results).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_NUM_HIDDENS, HP_ACT_FUN, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    hparams = {
        HP_NUM_UNITS: HP_NUM_UNITS.domain.values,
        HP_NUM_HIDDENS: HP_NUM_HIDDENS.domain.values,
        HP_ACT_FUN: HP_ACT_FUN.domain.values,
        # HP_DROPOUT: HP_DROPOUT.domain.values,
        HP_OPTIMIZER: HP_OPTIMIZER.domain.values,
    }
    return hparams
'''

''' TODO: Input: haparams
def train_test_model(hparams, logdir, x_train, y_train, x_val, y_val):
    model = tf.keras.models.Sequential()
    x = MaxPooling1D()(x_train)
    for i in range(hparams[HP_NUM_HIDDENS]):
        # model.add(Dense(hparams[HP_NUM_UNITS], activation = hparams[HP_ACT_FUN]))
        x = Dense(hparams[HP_NUM_UNITS], activation = hparams[HP_ACT_FUN])(x)

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, 
        # batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = (x_val, y_val),
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
        ]
    ) # Run with 1 epoch to speed things up for demo purposes

    # TODO: change evaluation matrix, using different functions
    _, accuracy = model.evaluate(x_val, y_val)

    return accuracy
'''


def apply_pooling(data, pool_size=1):
    data = np.reshape(data, [len(data), len(data[0]), 1])
    output = MaxPool1D(pool_size=pool_size)(data)
    return np.squeeze(output)

def model_creation(hidden_layers, hidden_units, act_function, learning_rate, optimizer):
    model = tf.keras.models.Sequential()
    # model.add(MaxPooling1D(5))
    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation = act_function))
    model.add(Dense(2))

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model

def run_yesno_training(model, x_train, y_train, pool_size, batch_size, logdir):
    if pool_size != 1:
        x_train = apply_pooling(x_train, pool_size)

    model.fit(x_train, y_train, 
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_split = VALIDATION_SPLIT,
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #     hp.KerasCallback(logdir, hparams),  # log hparams
        ]
    )
    model.summary()

    return model
