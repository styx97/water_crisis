""" 
    Filename: 
    Description: 
    Author: Sayantan Mahinder
    Date: 11/25/19
    
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import transformers
from transformers import BertModel, BertForSequenceClassification, AdamW, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

from tqdm import *
import math, os, sys, logging
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

import logging
import math
import os
import sys

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


# The input to classifier is cosine and (1 - cosine) weighted
class BertForAttribution(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForAttribution, self).__init__(config)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.sigmoid_weight = 2.0
        self.sig = nn.Sigmoid()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, 1)
        self.cos_w = 1.0
        self.init_weights()

    def forward(self, input_ids1=None, attention_mask1=None, input_ids2=None, attention_mask2=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs1 = self.bert(input_ids1,
                             attention_mask=attention_mask1,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds)

        outputs2 = self.bert(input_ids2,
                             attention_mask=attention_mask2,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds)

        # outputs[0] are of shape batch_size, sequence_length, hidden_size
        sentence_vector = outputs1[0]
        attribution_vector = outputs2[0]
        attribution_vector = torch.mean(attribution_vector, dim=1).unsqueeze(dim=1)
        cosine_vector = self.cos(sentence_vector, attribution_vector)
        reverse_cosine_vector = torch.ones_like(cosine_vector) - (self.cos_w * cosine_vector)

        topical_probs = self.sig(10*(cosine_vector - 0.5 * torch.ones_like(cosine_vector)))
        context_probs = self.sig(10 * reverse_cosine_vector)

        topic_vector = torch.mean(topical_probs[:, :, None] * sentence_vector, dim=1)
        context_vector = torch.mean(context_probs[:, :, None] * sentence_vector, dim=1)

        topic = torch.cat((topic_vector, context_vector), dim=1)
        topic_dropout = self.dropout(topic)
        logits = self.classifier(topic_dropout)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)
        else:
            outputs = (None, logits)
        return outputs

