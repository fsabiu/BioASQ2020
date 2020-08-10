import sys
import torch
import math  
from queue import PriorityQueue 
import os
from itertools import groupby
from operator import itemgetter
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, BertForQuestionAnswering, BertModel, BertForPreTraining
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_data, find_sub_list
from utils.evaluate import evaluate_factoid

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


def encode_daset(dataset_path, max_len, tokenizer, test_execution=-1):
    ##########################
    # Gli snippet che sono più lunghi di maxlen vengono scartati
    #########################

    print("...Encode dataset")
    if(test_execution!=-1):
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
       
        #sample[0] domanda
        #sample[1] risposta
        #sample[2] snippet

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

            # Domanda e snippet
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Risposta
            answer_token=tokenizer.encode(sample[1][0])
            answer_encoding = tokenizer.convert_ids_to_tokens(answer_token)
            
            #Ricerca della risposta nello snippet
            start,end=find_sub_list(all_tokens,answer_encoding)

            #Se la risposta non è all'interno dello snippet non viene considerata
            if(start!=-1):
                answer_list.append(sample[1])
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


def run_factoid_training(model, x_data, y_data, x_data_val, y_data_val, epochs, batch_size):
    print("...Training")
    model.fit(
        x_data,
        y_data,
        batch_size=batch_size,
        validation_data=(x_data_val, y_data_val),
        epochs=epochs,
        verbose=1,
    )
    return model

def extract_answer(start_scores, end_scores, all_tokens):
    #Estrazione delle 5 risposte più probabili
    #TODO: gestire il numero di combinazioni, con troppe le combinazioni sono troppe, inserire un threashold
    results_array=KMaxCombinations(start_scores,end_scores,5)
    final_answers=[]

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
                # Non dovrebbe servire eliminarli una volta che il modello è allenato bene, ma per questione di pulizia si eliminano anche questi token extra
                if(all_tokens[i] != "[PAD]" and all_tokens[i] != "[SEP]"):
                    answer += ' ' + all_tokens[i]
        final_answers.append((answer,float(elem[0])))
    return final_answers

def KMaxCombinations( arr1, arr2, K): 
    # Somma le combinaziooni di end e start per ottenere inizio e fine più probabili
    #Vedi test_list model per i dettagli

    dim=len(arr1)
    pq = PriorityQueue()

    # insert all the possible   
    # combinations in max heap. 
    for i in range(0, dim): 
        for j in range(0, dim): 
           a = arr1[i] + arr2[j]  
           pq.put((-a, a,i,j))

              
    # pop first N elements 
    counter = 0
    results_array=[]
    while (counter < K):
        elem=pq.get()
        #SI inseriscono solo le ripsoste in cui l'indice di inizio inizio è precedente all'indice di fine
        if(elem[2]<elem[3]):
            results_array.append((elem[1],elem[2],elem[3]))
            counter = counter + 1
    #Vengono restituite lo score, lo start e l'end
    return results_array

def merge_answer(predicted):
    
    sorter = sorted(predicted, key=itemgetter(1))
    grouper = groupby(sorter, key=itemgetter(1))

    res = {k: list(map(itemgetter(0), v)) for k, v in grouper}
    final_predicted=[]
    for key in res:
        complete_list=list(itertools.chain.from_iterable(res[key]))
        final_predicted.append(sorted(complete_list, key=lambda t: t[1], reverse=True)[:5])
    return final_predicted

def test_factoid_model(trained_model, tokenizer, x_data_test, answer_list):
    print("...Evaluate")

    predicted = []
    start_scores, end_scores = trained_model(x_data_test)
    last_elem=""
    last_elem_count=0
    merge_answer_list=[]

    for i in range(len(start_scores)):
        all_tokens = tokenizer.convert_ids_to_tokens(x_data_test[0][i])
        if(x_data_test[0][i]!=last_elem):
            last_elem_count+=1
            last_elem=x_data_test[0][i]
            merge_answer_list.append(answer_list[i])

        # Print risposta raw
        #answer = ' '.join(all_tokens[tf.argmax(start_scores[i]) : tf.argmax(end_scores[i])+1])
        # print(answer)

        answer_extract = extract_answer(start_scores[i], end_scores[i], all_tokens)

        predicted.append((answer_extract,last_elem_count))

    predicted=merge_answer(predicted)
    print(evaluate_factoid(predicted=predicted, target=merge_answer_list))
