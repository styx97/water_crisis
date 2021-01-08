import json
import os
from transformers import BertModel, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from bert_preprocess import preprocess_bert, preprocess_bert_indian
from bertForAttribution import BertForAttribution

path_to_dataset = ""   # insert dataset path (after chaining) 
train_indices = ""    # insert training indices in a plaintext file (newline separated)  
val_indices = ""    # insert training indices in a plaintext file (newline separated)

#Note : Indices must correspond to those of the dataset


def model_trainer_v2(path_to_dataset, train_indices_filepath, val_indices_filepath, fold=None):
    train_d, valid_d = preprocess_bert(path_to_dataset, train_indices_filepath, val_indices_filepath)

    input_ids_text, input_ids_topic, labels, attention_masks_text, attention_masks_topic = train_d
    input_ids_text_valid, input_ids_topic_valid, labels_valid, attention_masks_text_valid, attention_masks_topic_valid = valid_d
    print(len(input_ids_text), len(input_ids_topic), len(labels), len(attention_masks_text), len(attention_masks_topic))

    """
    Turn Everything into torch Tensors 
    """

    train_inputs_text = torch.LongTensor(input_ids_text)
    validation_inputs_text = torch.LongTensor(input_ids_text_valid)

    train_masks_text = torch.LongTensor(attention_masks_text)
    validation_masks_text = torch.LongTensor(attention_masks_text_valid)

    train_inputs_topic = torch.LongTensor(input_ids_topic)
    validation_inputs_topic = torch.LongTensor(input_ids_topic_valid)

    train_masks_topic = torch.LongTensor(attention_masks_topic)
    validation_masks_topic = torch.LongTensor(attention_masks_topic_valid)

    train_labels = torch.FloatTensor(labels)
    validation_labels = torch.FloatTensor(labels_valid)

    # Select a batch size for training. For fine-tuning BERT on a specific task,
    # the authors recommend a batch size of 16 or 32
    batch_size = 4  # select larger batch size if gpu is available

    # Dataloader for train
    train_data = TensorDataset(train_inputs_text, train_masks_text, train_inputs_topic, train_masks_topic, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # dataloader for eval
    validation_data = TensorDataset(validation_inputs_text, validation_masks_text, validation_inputs_topic,
                                    validation_masks_topic, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    print("Size of train_data ", len(train_data))
    print("Size of validation data ", len(validation_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model will run on device: {device}')
    # initialize the model -
    model = BertClassification.from_pretrained("bert-base-uncased")
    model.to(device)
    # print("stuff so that the full model is not printed")

    # decaying the weights of only certain layers to prevent overfitting
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # TO - DO - save best model
    # create model directory
    model_dir = '/home/sayantan/Desktop/projects/ganga-tripadvisor/final_training/bert_v5'
    if fold:
        model_dir = f'/home/sayantan/Desktop/projects/ganga-tripadvisor/final_training/bert_simple_kfold_v5/kfold_fold_{fold}'
    if not os.path.exists(model_dir):
        # os.rmdir(model_dir)
        os.mkdir(model_dir, mode=0o777)

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=1.4e-5,
                      eps=1e-06,
                      )
    optimizer.zero_grad()
    train_loss_set = []
    eval_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 5
    best_loss = float('inf')
    prev_model = None
    # trange is a tqdm wrapper around the normal python range
    for epoch_id in range(epochs):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to device
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_text_ids, b_text_mask, b_topic_ids, b_topic_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            output = model(
                input_ids1=b_text_ids,
                attention_mask1=b_text_mask,
                input_ids2=b_topic_ids,
                attention_mask2=b_topic_mask,
                labels=b_labels)
            loss = output[0]

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_text_ids.size(0)
            nb_tr_steps += 1
        train_epoch_loss = tr_loss / nb_tr_steps
        train_loss_set.append(tr_loss / nb_tr_steps)
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_text_ids, b_text_mask, b_topic_ids, b_topic_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                op = model(
                    input_ids1=b_text_ids,
                    attention_mask1=b_text_mask,
                    input_ids2=b_topic_ids,
                    attention_mask2=b_topic_mask,
                    labels=b_labels)

                loss_eval_batch = op[0]

            eval_loss += loss_eval_batch.item()
            # Move logits and labels to CPU
            # logits = logits.detach().cpu().numpy()
            # label_ids = b_labels.to('cpu').numpy()

            # tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        epoch_valid_loss = eval_loss / nb_eval_steps
        print("Validation Loss: {}".format(epoch_valid_loss))

        if prev_model and epoch_valid_loss > min(eval_loss_set):
            model = prev_model
            print("## Inflection in validation loss detected")
            break
        eval_loss_set.append(epoch_valid_loss)
        # TO-DO - Save the best models
        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss

        print("Saving model ")

        model_path = model_dir + '/{}_{}.model'.format(fold if fold else 'model', epoch_id)
        torch.save({
            'train_loss': train_epoch_loss,
            'val_loss': epoch_valid_loss,
            'epoch': epoch_id,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_path)
        prev_model = model

    history = {
        "train_loss_set": train_loss_set,
        "eval_loss_set": eval_loss_set  # size = number of epochs
    }
    with torch.no_grad():
        torch.cuda.empty_cache()
    return model, history


def model_trainer_indian(path_to_dataset, train_indices_filepath, val_indices_filepath, fold=None, run=None):
    train_d, valid_d = preprocess_bert_indian(path_to_dataset, train_indices_filepath, val_indices_filepath)

    input_ids_text, input_ids_topic, labels, attention_masks_text, attention_masks_topic = train_d
    input_ids_text_valid, input_ids_topic_valid, labels_valid, attention_masks_text_valid, attention_masks_topic_valid = valid_d
    print(len(input_ids_text), len(input_ids_topic), len(labels), len(attention_masks_text), len(attention_masks_topic))

    """
    Turn Everything into torch Tensors 
    """

    train_inputs_text = torch.LongTensor(input_ids_text)
    validation_inputs_text = torch.LongTensor(input_ids_text_valid)

    train_masks_text = torch.LongTensor(attention_masks_text)
    validation_masks_text = torch.LongTensor(attention_masks_text_valid)

    train_inputs_topic = torch.LongTensor(input_ids_topic)
    validation_inputs_topic = torch.LongTensor(input_ids_topic_valid)

    train_masks_topic = torch.LongTensor(attention_masks_topic)
    validation_masks_topic = torch.LongTensor(attention_masks_topic_valid)

    train_labels = torch.FloatTensor(labels)
    validation_labels = torch.FloatTensor(labels_valid)

    # Select a batch size for training. For fine-tuning BERT on a specific task,
    # the authors recommend a batch size of 16 or 32
    batch_size = 4  # select larger batch size if gpu is available

    # Dataloader for train
    train_data = TensorDataset(train_inputs_text, train_masks_text, train_inputs_topic, train_masks_topic, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # dataloader for eval
    validation_data = TensorDataset(validation_inputs_text, validation_masks_text, validation_inputs_topic,
                                    validation_masks_topic, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    print("Size of train_data ", len(train_data))
    print("Size of validation data ", len(validation_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model will run on device: {device}')
    # initialize the model -
    model = BertForAttribution21.from_pretrained("/home/sayantan/Desktop/projects/ganga-tripadvisor/data/indian_bert")
    model.to(device)
    # print("stuff so that the full model is not printed")

    # decaying the weights of only certain layers to prevent overfitting
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # TO - DO - save best model
    # create model directory
    model_dir = '/home/sayantan/Desktop/projects/ganga-tripadvisor/final_training/indian_bert_v5_fold_4'
    if fold:
        model_dir = f'/home/sayantan/Desktop/projects/ganga-tripadvisor/final_training/indian_bert_kfold_v5/kfold_fold_{fold}_{run}'
    if not os.path.exists(model_dir):
        # os.rmdir(model_dir)
        os.mkdir(model_dir, mode=0o777)

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=1.4e-5,
                      eps=1e-06,
                      )
    optimizer.zero_grad()
    train_loss_set = []
    eval_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 5
    best_loss = float('inf')
    prev_model = None
    # trange is a tqdm wrapper around the normal python range
    for epoch_id in range(epochs):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to device
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_text_ids, b_text_mask, b_topic_ids, b_topic_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            output = model(
                input_ids1=b_text_ids,
                attention_mask1=b_text_mask,
                input_ids2=b_topic_ids,
                attention_mask2=b_topic_mask,
                labels=b_labels)
            loss = output[0]

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_text_ids.size(0)
            nb_tr_steps += 1
        train_epoch_loss = tr_loss / nb_tr_steps
        train_loss_set.append(tr_loss / nb_tr_steps)
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_text_ids, b_text_mask, b_topic_ids, b_topic_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                op = model(
                    input_ids1=b_text_ids,
                    attention_mask1=b_text_mask,
                    input_ids2=b_topic_ids,
                    attention_mask2=b_topic_mask,
                    labels=b_labels)

                loss_eval_batch = op[0]

            eval_loss += loss_eval_batch.item()
            # Move logits and labels to CPU
            # logits = logits.detach().cpu().numpy()
            # label_ids = b_labels.to('cpu').numpy()

            # tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        epoch_valid_loss = eval_loss / nb_eval_steps
        print("Validation Loss: {}".format(epoch_valid_loss))

        if prev_model and epoch_valid_loss > min(eval_loss_set):
            model = prev_model
            print("## Inflection in validation loss detected")
            break
        eval_loss_set.append(epoch_valid_loss)
        # TO-DO - Save the best models
        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss

        print("Saving model ")

        model_path = model_dir + '/{}_{}.model'.format(fold if fold else 'model', epoch_id)
        torch.save({
            'train_loss': train_epoch_loss,
            'val_loss': epoch_valid_loss,
            'epoch': epoch_id,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_path)
        prev_model = model

    history = {
        "train_loss_set": train_loss_set,
        "eval_loss_set": eval_loss_set  # size = number of epochs
    }
    with torch.no_grad():
        torch.cuda.empty_cache()
    return model, history


def k_fold_training():
    # k_fold_indices = # Newline separated index files for kfold. 
    results = {}
    for fold in range(0, 5):

        train_file = k_fold_indices + '/' + 'fold_' + str(fold) + '_train.txt'
        valid_file = k_fold_indices + '/' + 'fold_' + str(fold) + '_val.txt'
        test_file = k_fold_indices + '/' + 'fold_' + str(fold) + '_test.txt'
        print(f'##########Training models for fold {fold}')
        best_model = None
        best_history = None
        for i in range(1):
            print(f'##########Iteration: {i}')
            model, history = model_trainer_v2(path_to_dataset, train_file, test_file, str(fold))
            if best_model and min(best_history['eval_loss_set']) < min(history['eval_loss_set']):
                continue
            best_model = model
            best_history = history
        print(f'##########Model Training for fold {fold} result: \n {best_history}')
        print(f'##########Testing models for fold {fold}')
        # # # # # # # # #
        # Remember to take care of the tokenizer
        # # # # # # # # #
        res = test_model_by_model(path_to_dataset, best_model, test_file, device=None, threshold=0.9, topN=3,
                                  if_debug=False, indian_tokenizer=None)
        results[fold] = res
    with open(k_fold_indices + '/' + 'fold_results_simple_bert.json', mode='w') as fw:
        json.dump(results, fw)


if __name__ == '__main__':
    model_trained, model_history = model_trainer_indian(path_to_dataset, train_indices, val_indices)
    print(model_history)
    # k_fold_training()
