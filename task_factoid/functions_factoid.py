import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, BertForQuestionAnswering, BertModel, BertForPreTraining
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_data

def model_creation(max_len, encoder):
    print("...Model creation")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
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

def prepare_dataset(dataset_raw):
    
    cleaned_dataset=[]
    for sample in dataset_raw:
        # sample[0]->Question
        # sample[1]->Answer
        # sample[2]->Snippets
        for snippet in sample[2]:
            entry=[sample[0],snippet]
            cleaned_dataset.append(entry)
    return cleaned_dataset

def encode_daset(dataset_path,max_len, tokenizer):
    print("...Encode dataset")
    dataset_raw = load_data(dataset_path, "factoid")[0:2]
    dataset_processed=prepare_dataset(dataset_raw)

    input_ids_list=[]
    token_type_ids_list=[]
    attention_mask_list=[]
    start_aswer_list=[]
    end_answer_list=[]

    for sample in dataset_processed:

        encoding = tokenizer.encode_plus(sample[0], sample[1])
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)


        # Aggiunta manuale start token end token
        start_aswer_list.append(4)
        end_answer_list.append(5)
        
        #Altre info
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        
    input_ids_list=np.array(input_ids_list)
    token_type_ids_list=np.array(token_type_ids_list)
    attention_mask_list=np.array(attention_mask_list)
    start_aswer_list=np.array(start_aswer_list)
    end_answer_list=np.array(end_answer_list)

    x_data=[input_ids_list,token_type_ids_list,attention_mask_list]
    y_data=[start_aswer_list,end_answer_list]
    return x_data,y_data
def run_factoid_training(model,x_data,y_data,epochs):
    print("...Training")
    model.fit(
    x_data,
    y_data,
    epochs=epochs,
    verbose=1,
    )