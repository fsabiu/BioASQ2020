import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluate import evaluate_factoid
from utils.data import load_data, find_sub_list
import time
import torch
import math
from queue import PriorityQueue
from itertools import groupby
from operator import itemgetter
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, BertForQuestionAnswering, BertModel, BertForPreTraining

def model_creation(max_len, learning_rate,encoder):
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
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model

def encode_dataset(dataset_path, max_len, tokenizer, test_execution=-1):
    ##########################
    # Gli snippet che sono più lunghi di maxlen vengono scartati
    # Con test_execution si mettono un numero di snippet minore del totale, in modo da eseguire test rapidi
    #########################

    print("...Encode dataset")
    if(test_execution != -1):
        ##
        dataset_processed = load_data(
            dataset_path, "factoid", singleSnippets=True)[0:test_execution]
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

        # sample[0] question
        # sample[1] answer
        # sample[2] snippet

        # Encoding della domanda e dello snippet
        encoding = tokenizer.encode_plus(sample[0], sample[2])
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]

        if(len(input_ids) < max_len):

            attention_mask = [1] * len(input_ids)
            padding_length = max_len - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)

            # Question and snippet
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Risposta
            answer_token = tokenizer.encode(sample[1][0])
            answer_encoding = tokenizer.convert_ids_to_tokens(answer_token)

            # Ricerca della risposta nello snippet
            start, end = find_sub_list(all_tokens, answer_encoding)
            
            # Se la risposta non è all'interno dello snippet non viene considerata
            if(start != -1):
                answer_list.append(sample[1][0])
                start_aswer_list.append(start)
                end_answer_list.append(end)

                # Altre info

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

    input_ids_list = np.array(input_ids_list)
    token_type_ids_list = np.array(token_type_ids_list)
    attention_mask_list = np.array(attention_mask_list)
    start_aswer_list = np.array(start_aswer_list)
    end_answer_list = np.array(end_answer_list)

    x_data = [input_ids_list, token_type_ids_list, attention_mask_list]
    y_data = [start_aswer_list, end_answer_list]

    return x_data, y_data, answer_list


def run_factoid_training(model, x_data, y_data, epochs, batch_size):
    #La validazione è fatta in automatico con validation split
    print("...Training")
    model.fit(
        x_data,
        y_data,
        batch_size=batch_size,
        validation_data=(x_data, y_data),#mettere validation split quando si fa un allenamento completo
        epochs=epochs,
        verbose=1,
        use_multiprocessing=True
    )
    return model

def test_factoid_model(trained_model, tokenizer, x_data_test, answer_list):
    print("...Evaluate")

    predicted = []
    start_scores, end_scores = trained_model(x_data_test)
    last_elem = ""
    last_elem_count = 0
    merge_answer_list = []

    for i in range(len(x_data_test[0])):
        all_tokens = tokenizer.convert_ids_to_tokens(x_data_test[0][i])

        if(answer_list[i]!=last_elem):
            last_elem_count += 1
            last_elem=answer_list[i]
            merge_answer_list.append(answer_list[i])

        # Print risposta raw
        #answer = ' '.join(all_tokens[tf.argmax(start_scores[i]) : tf.argmax(end_scores[i])+1])
        # print(answer)

        answer_extract = extract_answer(
            start_scores[i], end_scores[i], all_tokens,x_data_test[1][i])

        predicted.append((answer_extract, last_elem_count))


    predicted = merge_answer(predicted)

    predicted_cleaned=[[elem for elem,score in question] for question in predicted]

    print(predicted_cleaned)
    print(merge_answer_list)
    print(evaluate_factoid(predicted=predicted, target=merge_answer_list))

def extract_answer(start_scores, end_scores, all_tokens,token_type_ids):
    # Estrazione delle 5 risposte più probabili
    ##########
    print("start combination")
    start=time.time()
    ########
    results_array = KMaxCombinations(start_scores, end_scores, 5,token_type_ids)
    ##########
    end=time.time()
    print("Get top 5 compination:",end-start)
    #########
    final_answers = []

    for elem in results_array:
        answer_start = elem[1]
        answer_end = elem[2]
        answer = all_tokens[answer_start]
        # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):

            # If it's a subword token, then recombine it with the previous token.
            if all_tokens[i][0:2] == '##':
                answer += all_tokens[i][2:]

            # Otherwise, add a space then the token.
            else:
                # Non dovrebbe servire eliminarli una volta che il modello è allenato bene(non dovrebbe mai prendere i token pad), ma per questione di pulizia si eliminano anche questi token extra
                if(all_tokens[i] != "[PAD]" and all_tokens[i] != "[SEP]"):
                    answer += ' ' + all_tokens[i]
        final_answers.append((answer, float(elem[0])))
    return final_answers


def KMaxCombinations(start, end, K, token_type_ids):
    # Somma le combinaziooni di end e start per ottenere inizio e fine più probabili
    # Vedi test_list model per i dettagli

    dim = len(start)
    pq = PriorityQueue()

    # insert all the possible
    # combinations in max heap.
    for i in range(0, dim):
        for j in range(0, dim):
            if(i<=j and token_type_ids[i]==1 and token_type_ids[j]==1):
                a = start[i] + end[j]
                pq.put((-a, a, i, j))

    # pop first N elements
    counter = 0
    results_array = []
    while (counter < K):
        elem = pq.get()
        # SI inseriscono solo le ripsoste in cui l'indice di inizio inizio è precedente all'indice di fine
        results_array.append((elem[1], elem[2], elem[3]))
        counter = counter + 1
    # Vengono restituite lo score, lo start e l'end
    return results_array


def merge_answer(predicted):
    # Fa il merge delle risposte per la singola domanda, e restituisce le migliori 5

    sorter = sorted(predicted, key=itemgetter(1))
    grouper = groupby(sorter, key=itemgetter(1))

    res = {k: list(map(itemgetter(0), v)) for k, v in grouper}
    final_predicted = []
    for key in res:
        complete_list = list(itertools.chain.from_iterable(res[key]))
        final_predicted.append(
            sorted(complete_list, key=lambda t: t[1], reverse=True)[:5])
    return final_predicted



