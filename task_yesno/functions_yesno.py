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

VALIDATION_SPLIT = 0.15
EPOCHS = 500


def evaluate_yes_no(predicted, target):
    accuracy=accuracy_score(y_true=target, y_pred=predicted)

    f1_y=f1_score(y_true=target, y_pred=predicted, average='binary', pos_label=1)
    f1_n=f1_score(y_true=target, y_pred=predicted, average='binary', pos_label=0)
    macro_average_f_measure=(f1_y+f1_n)/2

    return {'accuracy':accuracy,'f1_yes':f1_y, 'f1_no':f1_n ,'macro_average_f_measure':macro_average_f_measure}

def get_embedding(data = None, file_embedding = None):
    if file_embedding is None:
        embeddings_elmo_pubmed = ELMoEmbeddings('pubmed') 
        pooling_model = DocumentPoolEmbeddings([embeddings_elmo_pubmed])
        embeddings = generate_embeddings_yesno_pooling(pooling_model, data, "embedding_yes_no.emb")
    else:
        embeddings = load_embeddings(file_embedding)
    return embeddings


def enconde_dataset(embeddings, pool_size=1, test_size=None):
    # Data includes questions (0) and snippets (2)
    data = np.array([np.concatenate((np.array(el[0]), np.array(el[2])),axis=None) for el in embeddings])
    labels = np.array([el[1] for el in embeddings])

    if test_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
        x_train = apply_pooling(x_train, pool_size)
        x_test = apply_pooling(x_test, pool_size)
        return x_train, y_train, x_test, y_test
    else: 
        return data, labels

def apply_pooling(data, pool_size=1):
    data = np.reshape(data, [len(data), len(data[0]), 1])
    output = MaxPool1D(pool_size=pool_size)(data)
    return np.squeeze(output)

def model_creation(hidden_layers, hidden_units, act_function, learning_rate, optimizer, size=None):
    model = tf.keras.models.Sequential()
    if size is not None:
        model.add(tf.keras.Input(shape=(size,)))

    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation = act_function))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    return model

def run_yesno_training(model, x_train, y_train, pool_size, batch_size, logdir):
    if pool_size != 1:
        x_train = apply_pooling(x_train, pool_size)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode = "min", min_delta = 0.001, patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, 
        batch_size = batch_size,
        epochs = EPOCHS,
        validation_split = VALIDATION_SPLIT,
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            earlystop_callback
        ]
    )

    if False:
        model.save_weights(logdir+"final_model", save_format="h5")

    model.summary()

    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict_classes(x_test)
    # print("Predicted: ", y_pred)
    # print("Corrected: ", y_test)
    count = 0
    for i, e in enumerate(y_test):
        if y_pred[i] != e:
            count+=1
    print("Wrong answers: ", count, "/", len(y_test))

    evaluation = evaluate_yes_no(y_pred, y_test)
    print(evaluation)
    return evaluation

def test_final_model(model, test_path):
    if False:
        data = load_data(test_path, "yesno")
        embeddings_elmo_pubmed = ELMoEmbeddings('pubmed') 
        pooling_model = DocumentPoolEmbeddings([embeddings_elmo_pubmed])
        test_data_embedded = generate_embeddings_yesno_pooling(pooling_model, data, "embedding_test.emb")
    else:
        test_data_embedded = load_embeddings('embedding_test.emb')

    data = np.array([np.concatenate((np.array(el[0]), np.array(el[2])),axis=None) for el in test_data_embedded])
    labels = np.array([el[1] for el in test_data_embedded])

    final_eval = evaluate_model(model, data, labels)
    return final_eval

