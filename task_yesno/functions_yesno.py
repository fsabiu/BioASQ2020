import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from flair.data import Sentence
import progressbar
import pickle
import sys, os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_data, generate_embeddings_yesno, generate_embeddings_yesno_pooling, load_embeddings


from flair.embeddings import ELMoEmbeddings, DocumentPoolEmbeddings
# from flair.embeddings import Sentence
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, InputLayer
from tensorflow.keras.models import Model
import numpy as np
VALIDATION_SPLIT = 0.33
EPOCHS = 5


def get_embedding(data, file_embedding = None):
    if file_embedding is None:
        embeddings_elmo_pubmed = ELMoEmbeddings('pubmed') 
        pooling_model = DocumentPoolEmbeddings([embeddings_elmo_pubmed])
        embeddings = generate_embeddings_yesno_pooling(pooling_model, data, "embedding_yes_no.emb")
    else:
        # ../data/embedding_yes_no.emb
        embeddings = load_embeddings(file_embedding)
    return embeddings


def enconde_dataset(embeddings):
    emb_numpy = np.array(embeddings)
    indices = np.arange(emb_numpy.shape[0])
    np.random.shuffle(indices)

    # Data includes questions (0) and snippets (2)
    data = emb_numpy[indices,0::2]
    labels = np.array(emb_numpy[indices,1], dtype=np.float)
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    data = np.array([np.concatenate([el[0].data, el[1].data]) for el in data])

    x_develop = data[:-nb_validation_samples]
    y_develop = labels[:-nb_validation_samples]

    x_train = x_develop[:-nb_validation_samples]
    x_train = np.reshape(x_train, [len(x_train), len(x_train[0]), 1])
    # x_train = np.expand_dims(x_train, 2)
    y_train = y_develop[:-nb_validation_samples]
    # y_train = np.expand_dims(y_train, 1)

    # Hold out validation
    x_val = x_develop[-nb_validation_samples:]
    y_val = y_develop[-nb_validation_samples:]

    # Hold out test
    x_test = data[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val, x_test, y_test

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

def model_creation(hidden_layers, hidden_units, act_function, x_train):

    model = tf.keras.models.Sequential()
    model.add(MaxPooling1D(5))
    # model.add(Conv1D(filters=5, kernel_size=5, padding="same"))
    for i in range(hidden_layers):
        # model.add(Dense(hparams[HP_NUM_UNITS], activation = hparams[HP_ACT_FUN]))
        model.add(Dense(hidden_units, activation = act_function))
    model.add(Dense(2))

    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

def run_yesno_training(model, x_train, y_train):
    print("training ...")
    model.fit(x_train, y_train, 
        # batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        # validation_data = (x_val, y_val),
        # callbacks = [
        #     tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #     hp.KerasCallback(logdir, hparams),  # log hparams
        # ]
    )

    model.summary()

    return model
    

dataset_path = "./data/training8b.json"
data = load_data(dataset_path, "yesno")
print("Getting embedding ")
emb = get_embedding(data, "./data/embedding_yes_no.emb")
print("Getting data")
x_train, y_train, x_val, y_val, x_test, y_test = enconde_dataset(emb)

training_model = model_creation(2, 5, 'sigmoid', x_train)
training_model = run_yesno_training(training_model, x_train, y_train)

print("Sample "+str(np.shape(x_test)))

y_pred = training_model.predict(x_test)
print(y_pred)
print(np.shape(y_test))


