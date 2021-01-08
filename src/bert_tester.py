from __future__ import absolute_import, division, print_function, unicode_literals
import json
import random
import transformers
from transformers import BertModel, BertForSequenceClassification, AdamW, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch
from bertForAttribution import BertForAttribution
import pandas as pd


def load_test_indices(test_index_file):
    """
    Function to load test indices from a file

    """

    test_index = []
    with open(test_index_file) as fp:
        for line in fp:
            test_index.append(int(line.strip()))
    return test_index


def load_model(bert_model_filepath, model_type):
    """
    Load a BERT model into the GPU from bert_model_filepath
    """
    model = model_type.from_pretrained("bert-base-uncased")
    checkpoint = torch.load(bert_model_filepath, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model_val_loss = checkpoint["val_loss"]
    print(f'Model Validation Loss: {model_val_loss}')
    model.eval()
    return model


def prediction_for_a_sentence(model, sentence, topic_names, device=None, indian_tokenizer=None):
    """
    Function to evaluate a single sentence against topic names
    Input :
        1. Model
        2. Sentence
        3. Names of the topics to be evaluated for attribution

    Output :
        1. A 2d array sorted in the order of matched probabilities  :
        Eg :
                [[0.8, "public water wastage"],
                 [0.7, "deforestation"],
                 ..
                ]
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_of_topics = len(topic_names)
    sentences = ["[CLS] " + sentence + " [SEP]"] * num_of_topics
    attribs = ["[CLS] " + attrib + " [SEP]" for attrib in topic_names]

    if indian_tokenizer:
        tokenizer = transformers.BertTokenizer.from_pretrained(
            '/home/sayantan/Desktop/projects/ganga-tripadvisor/data/indian_bert', do_lower_case=False)
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    tokenized_attribs = [tokenizer.tokenize(attrib) for attrib in attribs]

    MAX_LEN_SENT = 128
    MAX_LEN_TOPIC = 8
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

    # turn everything into tensors:
    inputs_text = torch.LongTensor(input_ids_text).to(device)
    masks_text = torch.LongTensor(attention_masks_text).to(device)
    inputs_topic = torch.LongTensor(input_ids_attribs).to(device)
    masks_topic = torch.LongTensor(attention_masks_topic).to(device)

    test_data = TensorDataset(inputs_text, masks_text, inputs_topic, masks_topic)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=num_of_topics)

    attributions = []
    for _, batch in enumerate(test_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_text_ids, b_text_mask, b_topic_ids, b_topic_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            op = model(
                input_ids1=b_text_ids,
                attention_mask1=b_text_mask,
                input_ids2=b_topic_ids,
                attention_mask2=b_topic_mask)
            logits = op[1]
            probs = list(torch.sigmoid(logits).squeeze())
            for ind, prob in enumerate(probs):
                attributions.append([prob, topic_names[ind]])

    attributions = sorted(attributions, key=lambda x: x[0], reverse=True)
    
    #convert tensor to float
    for element in attributions : 
        element[0] = element[0].item()

    df = pd.DataFrame(attributions, columns =['Topic Score', 'Factor'])
     
    return df


def test_model(model_type, path_to_dataset, bert_model_filepath, path_to_indices, device=None, threshold=0.92, topN=3,
               indian_tokenizer=None, if_debug=False):
    df = pd.read_csv(path_to_dataset)
    topic_names = sorted(list(set(list(df['attrib_words']))))
    test_indices = load_test_indices(path_to_indices)
    df = df[df.index.isin(test_indices)]
    print(f"The shape of the dataframe is: {df.shape}, topics: {len(topic_names)}")

    model = load_model(bert_model_filepath, model_type)
    with open('/media/styx97/My Passport/water_crisis/sheet_to_dataset/clean_files/v5/topic_sanity.json') as fp:
        multi_labels = json.load(fp)
    level1_correct = 0
    level2_top1_correct = 0
    level2_top3_correct = 0
    tp1 = fp1 = fn1 = 0
    tp2 = fp2 = fn2 = 0
    tp3 = fp3 = fn3 = 0
    k = 0

    for index, row in df.iterrows():
        k += 1
        sentence = row.sentences
        label = row.attribution
        labeled_topic = row.attrib_words
        result = prediction_for_a_sentence(model=model, sentence=sentence, topic_names=topic_names, device=device,
                                           indian_tokenizer=indian_tokenizer)

        if if_debug:
            print(
                f'{sentence}::{label}\tlabeled_topic: {labeled_topic}\ntop1_topic: {result[0][1]}\ttop1_score: {result[0][0]}'
                f'\ntop3_topic: {set([result[i][1] for i in range(topN)])}\ttop3_score{result[2][0]}\n\n')
        if label == 0:
            is_level1_correct = is_level2_top1_correct = is_level2_top3_correct = result[0][0] < threshold
            if not is_level1_correct:
                fp1 += 1
                fp2 += 1
                fp3 += 1
            level1_correct += 1 if is_level1_correct else 0
            level2_top1_correct += 1 if is_level2_top1_correct else 0
            level2_top3_correct += 1 if is_level2_top3_correct else 0
        else:
            is_level1_correct = result[0][0] >= threshold
            # is_level2_top1_correct = (result[0][0] >= threshold and result[0][1] == labeled_topic)
            is_level2_top1_correct = (result[0][0] >= threshold and (result[0][1] in multi_labels[sentence]))
            # is_level2_top3_correct = (
            #         result[0][0] >= threshold and (labeled_topic in set([result[i][1] for i in range(topN)])))
            is_level2_top3_correct = (result[0][0] >= threshold and any([x in set([result[i][1] for i in range(topN)])
                                                                         for x in multi_labels[sentence]]))

            if is_level1_correct:
                tp1 += 1
            else:
                fn1 += 1
            if is_level2_top1_correct:
                tp2 += 1
            else:
                fn2 += 1

            if is_level2_top3_correct:
                tp3 += 1
            else:
                fn3 += 1
            level1_correct += 1 if is_level1_correct else 0
            level2_top1_correct += 1 if is_level2_top1_correct else 0
            level2_top3_correct += 1 if is_level2_top3_correct else 0
        if if_debug:
            print(f'{k}:{level1_correct}:{level2_top1_correct}:{level2_top3_correct}')

    print(f"Level 1 tp and fp are {tp1} and {fp1}")
    precision1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f1_1 = 2 * precision1 * recall1 / (precision1 + recall1)

    print(f"Level 2 top 1 tp and fp are {tp2} and {fp2}")
    precision2 = tp2 / (tp2 + fp2)
    recall2 = tp2 / (tp2 + fn2)
    f1_2 = 2 * precision2 * recall2 / (precision2 + recall2)

    print(f"Level 2 top 3 tp and fp are {tp3} and {fp3}")
    precision3 = tp3 / (tp3 + fp3)
    recall3 = tp3 / (tp3 + fn3)
    f1_3 = 2 * precision3 * recall3 / (precision3 + recall3)

    print(f'Total: {len(test_indices)}, level1_correct: {level1_correct}, '
          f'level2_top1_correct: {level2_top1_correct}, level2_top3_correct: {level2_top3_correct}')
    print(
        f'Level1 Accuracy: {level1_correct / len(test_indices)} \nLevel2 Top1 Acc: {level2_top1_correct / len(test_indices)} '
        f'\nLevel2 Top3 Acc: {level2_top3_correct / len(test_indices)}')
    print(f'Level1: Precision: {precision1} Recall: {recall1} F1: {f1_1}')
    print(f'Level2_top1_correct: Precision: {precision2} Recall: {recall2} F1: {f1_2}')
    print(f'Level2_top3_correct: Precision: {precision3} Recall: {recall3} F1: {f1_3}')


def test_model_by_model(path_to_dataset, model, path_to_indices, device=None, threshold=0.92,
                        topN=3,
                        if_debug=False, indian_tokenizer=None):
    df = pd.read_csv(path_to_dataset)
    topic_names = sorted(list(set(list(df['attrib_words']))))
    test_indices = load_test_indices(path_to_indices)
    df = df[df.index.isin(test_indices)]
    print(f"The shape of the dataframe is: {df.shape}, topics: {len(topic_names)}")

    level1_correct = 0
    level2_top1_correct = 0
    level2_top3_correct = 0
    tp1 = fp1 = fn1 = 0
    tp2 = fp2 = fn2 = 0
    tp3 = fp3 = fn3 = 0
    k = 0

    with open('/media/styx97/My Passport/water_crisis/sheet_to_dataset/clean_files/v5/topic_sanity.json') as fp:
        multi_labels = json.load(fp)

    for index, row in df.iterrows():
        k += 1
        sentence = row.sentences
        label = row.attribution
        labeled_topic = row.attrib_words
        result = prediction_for_a_sentence(model=model, sentence=sentence, topic_names=topic_names,
                                           device=device, indian_tokenizer=indian_tokenizer)

        if if_debug:
            print(
                f'{sentence}::{label}\tlabeled_topic: {labeled_topic}\ntop1_topic: {result[0][1]}\ttop1_score: {result[0][0]}'
                f'\ntop3_topic: {set([result[i][1] for i in range(topN)])}\ttop3_score{result[2][0]}\n\n')
        if label == 0:
            is_level1_correct = is_level2_top1_correct = is_level2_top3_correct = result[0][0] < threshold
            if not is_level1_correct:
                fp1 += 1
                fp2 += 1
                fp3 += 1
            level1_correct += 1 if is_level1_correct else 0
            level2_top1_correct += 1 if is_level2_top1_correct else 0
            level2_top3_correct += 1 if is_level2_top3_correct else 0
        else:
            is_level1_correct = result[0][0] >= threshold
            # is_level2_top1_correct = (result[0][0] >= threshold and result[0][1] == labeled_topic)
            is_level2_top1_correct = (result[0][0] >= threshold and (result[0][1] in multi_labels[sentence]))
            # is_level2_top3_correct = (
            #         result[0][0] >= threshold and (labeled_topic in set([result[i][1] for i in range(topN)])))
            is_level2_top3_correct = (result[0][0] >= threshold and any([x in set([result[i][1] for i in range(topN)])
                                                                         for x in multi_labels[sentence]]))

            if is_level1_correct:
                tp1 += 1
            else:
                fn1 += 1
            if is_level2_top1_correct:
                tp2 += 1
            else:
                fn2 += 1

            if is_level2_top3_correct:
                tp3 += 1
            else:
                fn3 += 1
            level1_correct += 1 if is_level1_correct else 0
            level2_top1_correct += 1 if is_level2_top1_correct else 0
            level2_top3_correct += 1 if is_level2_top3_correct else 0
        if if_debug:
            print(f'{k}:{level1_correct}:{level2_top1_correct}:{level2_top3_correct}')

    print(f"Level 1 tp and fp are {tp1} and {fp1}")
    precision1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f1_1 = 2 * precision1 * recall1 / (precision1 + recall1)

    print(f"Level 2 top 1 tp and fp are {tp2} and {fp2}")
    precision2 = tp2 / (tp2 + fp2)
    recall2 = tp2 / (tp2 + fn2)
    f1_2 = 2 * precision2 * recall2 / (precision2 + recall2)

    print(f"Level 2 top 3 tp and fp are {tp3} and {fp3}")
    precision3 = tp3 / (tp3 + fp3)
    recall3 = tp3 / (tp3 + fn3)
    f1_3 = 2 * precision3 * recall3 / (precision3 + recall3)

    print(f'Total: {len(test_indices)}, level1_correct: {level1_correct}, '
          f'level2_top1_correct: {level2_top1_correct}, level2_top3_correct: {level2_top3_correct}')
    print(
        f'Level1 Accuracy: {level1_correct / len(test_indices)} \nLevel2 Top1 Acc: {level2_top1_correct / len(test_indices)} '
        f'\nLevel2 Top3 Acc: {level2_top3_correct / len(test_indices)}')
    print(f'Level1: Precision: {precision1} Recall: {recall1} F1: {f1_1}')
    print(f'Level2_top1_correct: Precision: {precision2} Recall: {recall2} F1: {f1_2}')
    print(f'Level2_top3_correct: Precision: {precision3} Recall: {recall3} F1: {f1_3}')
    return {'Level1': (precision1, recall1, f1_1),
            'Level2_top1': (precision2, recall2, f1_2),
            'Level2_top3': (precision3, recall3, f1_3),
            }


def test_on_unseen_data(model_type, bert_model_filepath, path_to_data_json, path_to_attrib, threshold=0.9, topN=3):
    df = pd.read_csv(path_to_attrib)
    topic_names = sorted(list(set(list(df['attrib_words']))))
    topic_example = {"None": []}
    topic_dist = {"None": 0}
    count = 0
    model = load_model(bert_model_filepath, model_type)
    for topic in topic_names:
        topic_dist[topic] = 0
        topic_example[topic] = []
    with open(path_to_data_json) as fp:
        data = json.load(fp)
        for id in data:
            count += 1
            sentence = ' '.join(data[id])
            result = prediction_for_a_sentence(model=model, sentence=sentence, topic_names=topic_names, device=None)
            top_attrib = result[0][1]
            top_score = result[0][0]
            if top_score >= threshold:
                topic_dist[top_attrib] += 1

                if len(topic_example[top_attrib]) >= 5:
                    index = random.randint(0, min(4, len(topic_example[top_attrib])))
                    topic_example[top_attrib][index] = sentence
                else:
                    topic_example[top_attrib].append(sentence)
            else:
                topic_dist["None"] += 1
                if len(topic_example["None"]) >= 5:
                    index = random.randint(0, min(4, len(topic_example["None"])))
                    topic_example["None"][index] = sentence
                else:
                    topic_example["None"].append(sentence)
            if count % 500 == 0:
                out_file = '/home/sayantan/Desktop/projects/ganga-tripadvisor/data/distrib.preprocessed_40k.json'
                with open(out_file, mode='w') as fw:
                    json.dump({"topic_dist": topic_dist, "topic_example": topic_example}, fw)
                print(f'completed {count} sentences')

    return topic_dist, topic_example


def test_kfold(fold_model_paths, model_type, path_to_dataset, k_fold_indices, device=None, threshold=0.92,
               topN=3, if_debug=False, indian_tokenizer=None):
    results = {}
    for fold in fold_model_paths:
        model_path = fold_model_paths[fold]
        model = load_model(model_path, model_type)
        test_file = k_fold_indices + '/' + 'fold_' + str(fold) + '_test.txt'
        res = test_model_by_model(path_to_dataset, model, test_file, device=None, threshold=0.9,
                                  topN=3, if_debug=False, indian_tokenizer=None)
        results[fold] = res
    with open('/media/styx97/My Passport/water_crisis/sheet_to_dataset/clean_files/v5/fold_results_indian_bert_2.json',
              mode='w') as fw:
        json.dump(results, fw)


if __name__ == '__main__':
    
    path_to_dataset = "/media/styx97/My Passport/water_crisis/sheet_to_dataset/clean_files/v5/chained_comments_replaced_5.csv"
    path_to_indices = "/media/styx97/My Passport/water_crisis/sheet_to_dataset/clean_files/v5/fold_0_test.txt"
    bert_model_filepath = "/home/styx97/Windows/backup/water/"
    # path_to_data_json = '/home/sayantan/Desktop/projects/ganga-tripadvisor/data/preprocessed_40k.json'
    # out_file = '/home/sayantan/Desktop/projects/ganga-tripadvisor/data/distrib.preprocessed_40k.json'
    #test_kfold(fold_model_paths, BertForAttribution21, path_to_dataset, k_fold_indices)
    
    test_model(BertForAttribution21, path_to_dataset, bert_model_filepath, path_to_indices, device=None, threshold=0.9,
               topN=3, indian_tokenizer=None, if_debug=True)

    # topic_dist, topic_example = test_on_unseen_data(BertForAttribution21, bert_model_filepath, path_to_data_json, path_to_dataset)
    #
    # print('topic dist')
    # print(topic_dist)

    # # Single sentence prediction
    # model = load_model(bert_model_filepath, BertForAttribution21)
    # sentence = "govt is not taking necessary steps to save groundwater for future"
    # # # # sentence = "India's large population is still lower than China"
    # # # #
    # # # # # sentence = "Cutting down trees is causing this water crisis"
    # # # # # sentence = "Cutting down trees will reduce the oxygen levels"
    # # # # # sentence = "Among all recent issues India was hit most by this water crisis"
    # # # # sentence = "Use of smartphones is more among young Indians"
    # words = sentence.split(' ')
    # print(f'num words: {len(words)}')
    # df = pd.read_csv(path_to_dataset)
    # topic_names = sorted(list(set(list(df['attrib_words']))))
    # print(topic_names)
    # # # # topic_names = ['technology']
    # print(prediction_for_a_sentence(model=model, sentence=sentence, topic_names=topic_names))
    # # # sentence = "India has a large population causing this water crisis"
    # # # sentence = "India has a large population causing this water crisis"
