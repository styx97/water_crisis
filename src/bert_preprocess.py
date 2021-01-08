import transformers
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import random
from operator import itemgetter
# from tqdm import *
import math, os, sys, logging
import pandas as pd
from pprint import pprint


def preprocess_bert(path_to_dataset, train_filepath, val_filepath):
    df = pd.read_csv(path_to_dataset)
    df = df[["sentences", "attribution", "attrib_words"]]
    # df = df[:-500]  # set aside the last 200 values for testing
    print("Shape of the input dataframe is ", df.shape)

    # stratified sampling for eval dataset :
    topic_names = set(list(df['attrib_words']))
    
    total_indices = df.index
    val_indices = []
    train_indices = [] 
    
    with open(val_filepath) as file:
        for line in file: 
            val_indices.append(int(line))
    
    with open(train_filepath) as file:
        for line in file: 
            train_indices.append(int(line))

#     #Hacky - remove later on 
#     val_indices = [x for x in val_indices if x in total_indices] 
#     train_indices = [x for x in train_indices if x in total_indices] 
    
    #assert (len(train_indices) + len(val_indices) == df.shape[0])

    # separate sentences and make pad them with delimiters
    sentences = df.sentences.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.attribution.values
    attribs = df.attrib_words.values
    attribs = ["[CLS] " + attrib + " [SEP]" for attrib in attribs]

    # tokenize the sentences - 
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    tokenized_attribs = [tokenizer.tokenize(attrib) for attrib in attribs]
    # print(tokenized_texts[0])

    MAX_LEN_SENT = 128
    MAX_LEN_TOPIC = 8

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary and pad tokens
    input_ids_text = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids_text = pad_sequences(input_ids_text, maxlen=MAX_LEN_SENT, dtype="long", truncating="post", padding="post")

    # same for topics ids - 
    input_ids_attribs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_attribs]
    input_ids_attribs = pad_sequences(input_ids_attribs, maxlen=MAX_LEN_TOPIC, dtype="long", truncating="post",
                                      padding="post")

    # Create attention masks
    attention_masks_text, attention_masks_topic = [], []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_text:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_text.append(seq_mask)

    for seq in input_ids_attribs:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_topic.append(seq_mask)

    # print(attention_masks[0])

    input_id_text_tr, input_id_attribs_tr, labels_tr, at_mask_text_tr, at_mask_topic_tr = [input_ids_text[i] for i in
                                                                                          train_indices], \
                                                                                          [input_ids_attribs[i] for i in
                                                                                           train_indices], \
                                                                                          [labels[i] for i in
                                                                                           train_indices], \
                                                                                          [attention_masks_text[i] for i
                                                                                           in train_indices], \
                                                                                          [attention_masks_topic[i] for
                                                                                           i in train_indices]

    input_id_text_val, input_id_attribs_val, labels_val, at_mask_text_val, at_mask_topic_val = [input_ids_text[i] for i
                                                                                                in val_indices], [
                                                                                                   input_ids_attribs[i]
                                                                                                   for i in
                                                                                                   val_indices], \
                                                                                               [labels[i] for i in
                                                                                                val_indices], [
                                                                                                   attention_masks_text[
                                                                                                       i] for i in
                                                                                                   val_indices], [
                                                                                                   attention_masks_topic[
                                                                                                       i] for i in
                                                                                                   val_indices]

    return (input_id_text_tr, input_id_attribs_tr, labels_tr, at_mask_text_tr, at_mask_topic_tr), \
           (input_id_text_val, input_id_attribs_val, labels_val, at_mask_text_val, at_mask_topic_val)


def preprocess_bert_indian(path_to_dataset, train_filepath, val_filepath):
    df = pd.read_csv(path_to_dataset)
    df = df[["sentences", "attribution", "attrib_words"]]
    # df = df[:-500]  # set aside the last 200 values for testing
    print("Shape of the input dataframe is ", df.shape)

    # stratified sampling for eval dataset :
    topic_names = set(list(df['attrib_words']))

    total_indices = df.index
    val_indices = []
    train_indices = []

    with open(val_filepath) as file:
        for line in file:
            val_indices.append(int(line))

    with open(train_filepath) as file:
        for line in file:
            train_indices.append(int(line))

    #     #Hacky - remove later on
    #     val_indices = [x for x in val_indices if x in total_indices]
    #     train_indices = [x for x in train_indices if x in total_indices]

    # assert (len(train_indices) + len(val_indices) == df.shape[0])

    # separate sentences and make pad them with delimiters
    sentences = df.sentences.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.attribution.values
    attribs = df.attrib_words.values
    attribs = ["[CLS] " + attrib + " [SEP]" for attrib in attribs]

    # tokenize the sentences -
    tokenizer = transformers.BertTokenizer.from_pretrained('/home/sayantan/Desktop/projects/ganga-tripadvisor/data/indian_bert', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    tokenized_attribs = [tokenizer.tokenize(attrib) for attrib in attribs]
    # print(tokenized_texts[0])

    MAX_LEN_SENT = 128
    MAX_LEN_TOPIC = 8

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary and pad tokens
    input_ids_text = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids_text = pad_sequences(input_ids_text, maxlen=MAX_LEN_SENT, dtype="long", truncating="post", padding="post")

    # same for topics ids -
    input_ids_attribs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_attribs]
    input_ids_attribs = pad_sequences(input_ids_attribs, maxlen=MAX_LEN_TOPIC, dtype="long", truncating="post",
                                      padding="post")

    # Create attention masks
    attention_masks_text, attention_masks_topic = [], []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_text:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_text.append(seq_mask)

    for seq in input_ids_attribs:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_topic.append(seq_mask)

    # print(attention_masks[0])

    input_id_text_tr, input_id_attribs_tr, labels_tr, at_mask_text_tr, at_mask_topic_tr = [input_ids_text[i] for i in
                                                                                           train_indices], \
                                                                                          [input_ids_attribs[i] for i in
                                                                                           train_indices], \
                                                                                          [labels[i] for i in
                                                                                           train_indices], \
                                                                                          [attention_masks_text[i] for i
                                                                                           in train_indices], \
                                                                                          [attention_masks_topic[i] for
                                                                                           i in train_indices]

    input_id_text_val, input_id_attribs_val, labels_val, at_mask_text_val, at_mask_topic_val = [input_ids_text[i] for i
                                                                                                in val_indices], [
                                                                                                   input_ids_attribs[i]
                                                                                                   for i in
                                                                                                   val_indices], \
                                                                                               [labels[i] for i in
                                                                                                val_indices], [
                                                                                                   attention_masks_text[
                                                                                                       i] for i in
                                                                                                   val_indices], [
                                                                                                   attention_masks_topic[
                                                                                                       i] for i in
                                                                                                   val_indices]

    return (input_id_text_tr, input_id_attribs_tr, labels_tr, at_mask_text_tr, at_mask_topic_tr), \
           (input_id_text_val, input_id_attribs_val, labels_val, at_mask_text_val, at_mask_topic_val)