from utils.evaluate import evaluate_factoid
from utils.data import load_data
import sys
import torch
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, BertForQuestionAnswering, BertModel, BertForPreTraining
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def model_creation(max_len, encoder):
    print("...Model creation")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
                        )[0]

    start_logits = layers.Dense(
        1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


def encode_daset(dataset_path, max_len, tokenizer, test_execution=False):

    print("...Encode dataset")
    if(test_execution):
        dataset_processed = load_data(
            dataset_path, "factoid", singleSnippets=True)[0:2]
    else:
        dataset_processed = load_data(
            dataset_path, "factoid", singleSnippets=True)

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    start_aswer_list = []
    end_answer_list = []
    answer_list = []

    for sample in dataset_processed:
        answer_list.append(sample[1])
        encoding = tokenizer.encode_plus(sample[0], sample[2])
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  #
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

        # Aggiunta manuale start token end token
        # TODO: Aggiungere la ricerca automatica nello snippet
        start_aswer_list.append(4)
        end_answer_list.append(5)

        # Altre info
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)

    input_ids_list = np.array(input_ids_list)
    token_type_ids_list = np.array(token_type_ids_list)
    attention_mask_list = np.array(attention_mask_list)
    start_aswer_list = np.array(start_aswer_list)
    end_answer_list = np.array(end_answer_list)

    # TODO:eliminare indici
    x_data = [input_ids_list[0:3],
              token_type_ids_list[0:3], attention_mask_list[0:3]]
    y_data = [start_aswer_list[0:3], end_answer_list[0:3]]
    answer_list = answer_list[0:3]
    return x_data, y_data, answer_list


def run_factoid_training(model, x_data, y_data, epochs):
    print("...Training")
    # TODO:Aggiungere gestione validazione
    model.fit(
        x_data,
        y_data,
        epochs=epochs,
        verbose=1,
    )
    return model


def extract_answer(start_scores, end_scores, all_tokens):
    # TODO: eliminare il token di padding
    answer_start = tf.argmax(start_scores)
    answer_end = tf.argmax(end_scores)
    answer = all_tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if all_tokens[i][0:2] == '##':
            answer += all_tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + all_tokens[i]

    # TODO:Gestire generazione 5 risposte
    return [answer, answer, answer, answer, answer]


def test_factoid_model(trained_model, tokenizer, x_data_test, answer_list):

    print("...Evaluate")

    predicted = []
    start_scores, end_scores = trained_model(x_data_test)

    for i in range(len(start_scores)):
        all_tokens = tokenizer.convert_ids_to_tokens(x_data_test[0][i])

        # Print risposta raw
        #answer = ' '.join(all_tokens[tf.argmax(start_scores[i]) : tf.argmax(end_scores[i])+1])

        answers = extract_answer(start_scores[i], end_scores[i], all_tokens)
        predicted.append(answers)

    print(evaluate_factoid(predicted=predicted, target=answer_list))
