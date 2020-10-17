import argparse
import copy
from itertools import chain
import numpy as np
import os
import pandas as pd
# from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_bart import shift_tokens_right
import yaml

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, ViggoDataset
from seq2seq.slot_aligner.slot_alignment import score_alignment
from seq2seq.task_config import TestConfig, TrainingConfig


def load_config(config_name, dataset_name, task, print_config=False):
    config_path = os.path.join('seq2seq', 'config', dataset_name, task, config_name + '.yaml')

    try:
        with open(config_path) as f_config:
            config = yaml.safe_load(f_config)
    except FileNotFoundError:
        print(f'Error: config file "{config_path}" not found')
        sys.exit()
    except yaml.YAMLError as err:
        print(err)
        sys.exit()

    if print_config:
        print(f'>> Starting a "{task}" task with the following parameters:')
        print(yaml.dump(config, default_flow_style=False))
        print()

    return config


def load_pretrained_bart_model_and_tokenizer(model_name, special_tokens=None):
    # Load pretrained tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    special_tokens = {
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    # Load pretrained model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_pretrained_gpt2_model_and_tokenizer(model_name, special_tokens=None):
    # Load pretrained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    special_tokens = {
        'bos_token': '<|begoftext|>',
        'pad_token': '<PAD>',
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_pretrained_t5_model_and_tokenizer(model_name, special_tokens=None):
    # Load pretrained tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    special_tokens = {
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    # Load pretrained model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_model_checkpoint(model, model_name, epoch, step):
    model_dir = os.path.join('seq2seq', 'model')
    if not os.path.exists(model_dir):
        raise NotADirectoryError('No saved checkpoint found')

    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    file_name = '{}_epoch_{}_step_{}.pt'.format(model_name, epoch, step)
    checkpoint_path = os.path.join(model_dir, file_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('Checkpoint "{}" not found'.format(file_name))

    model.load_state_dict(torch.load(checkpoint_path))


def save_training_config(config):
    config_dir = os.path.join('seq2seq', 'model', 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    file_name = 'training_config.yaml'
    with open(os.path.join(config_dir, file_name), 'w') as f_out:
        yaml.dump(config, f_out, default_flow_style=False)


def save_model(model, model_name, epoch, step):
    model_dir = os.path.join('seq2seq', 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    file_name = '{}_epoch_{}_step_{}.pt'.format(model_name, epoch, step)
    torch.save(model.state_dict(), os.path.join(model_dir, file_name))


def create_label_mask(input_ids, input_mask, label_mask):
    label_offsets = input_mask.sum(dim=1) - label_mask.sum(dim=1)

    # DEBUG
    # print('>> label offsets:', label_offsets)

    mask = torch.zeros_like(input_ids)
    mask[torch.arange(input_ids.shape[0]), label_offsets] = 1
    mask = 1 - mask.cumsum(dim=1)

    # DEBUG
    # print('>> label mask:', mask)

    return mask


def train(config, dataset_class, device='cpu'):
    train_loss_sum = 0.0
    steps_since_last_eval = 0

    # Load model and corresponding tokenizer
    if 'gpt2' in config.pretrained_model:
        loading_function = load_pretrained_gpt2_model_and_tokenizer
        is_enc_dec = False
    elif 'bart' in config.pretrained_model:
        loading_function = load_pretrained_bart_model_and_tokenizer
        is_enc_dec = True
    elif 't5' in config.pretrained_model:
        loading_function = load_pretrained_t5_model_and_tokenizer
        is_enc_dec = True
    else:
        print('Error: model "{}" not supported'.format(config.pretrained_model))
        sys.exit()

    model, tokenizer = loading_function(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    model = model.to(device)

    # Load training and validation data
    train_set = dataset_class(tokenizer, 'train', lowercase=True, convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec)
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_set = dataset_class(tokenizer, 'valid', lowercase=True, convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec)
    valid_data_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    valid_set_grouped = dataset_class(tokenizer, 'valid', lowercase=True, convert_slot_names=config.convert_slot_names,
                                      group_by_mr=True, separate_source_and_target=is_enc_dec)
    valid_grouped_data_loader = DataLoader(valid_set_grouped, batch_size=1, shuffle=False, num_workers=0)

    # Determine the training steps at which validation should be performed in each epoch
    eval_steps = np.delete(np.linspace(0, len(train_data_loader), config.eval_times_per_epoch + 1, dtype=int), 0)

    # Set up the optimizer and learning rate scheduler
    num_training_steps = len(train_data_loader) * config.num_epochs
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config.num_warmup_steps,
                                                num_training_steps=num_training_steps)

    if config.fp16:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, config.num_epochs + 1):
        print()
        print(' *************** ')
        print('**   EPOCH {:<2}  **'.format(epoch))
        print(' *************** ')
        print()

        for step, batch in enumerate(tqdm(train_data_loader, desc='Step'), start=1):
            # tokenizer.padding_side = 'left'

            if is_enc_dec:
                # inputs = tokenizer.prepare_seq2seq_batch(batch[0], batch[1], max_length=config.max_seq_length,
                #                                          padding=True, truncation=True, return_tensors='pt')
                inputs = tokenizer(batch[0], add_special_tokens=True, max_length=config.max_seq_length,
                                   padding=True, truncation=True, return_tensors='pt')
                targets = tokenizer(batch[1], add_special_tokens=True, max_length=config.max_seq_length,
                                    padding=True, truncation=True, return_tensors='pt')

                input_ids = inputs['input_ids']
                input_mask = inputs['attention_mask']
                label_ids = targets['input_ids'].clone()
                label_ids[label_ids == tokenizer.pad_token_id] = -100

                if 'bart' in config.pretrained_model:
                    """Prepare decoder inputs manually because BART gets confused by the -100 mask values during
                    automatic generation of decoder inputs from labels, expecting the padding token IDs instead."""
                    decoder_input_ids = shift_tokens_right(targets['input_ids'], tokenizer.pad_token_id)
                    # decoder_mask = targets['attention_mask']
                    decoder_input_tensor = decoder_input_ids.to(device)
                    # decoder_mask_tensor = decoder_mask.to(device)
                else:
                    # Decoder input IDs and mask are inferred automatically from labels
                    decoder_input_tensor = None
            else:
                # TODO: Experiment with the token_type_id parameter.
                inputs = tokenizer(batch[0], add_special_tokens=False, max_length=config.max_seq_length,
                                   padding=True, truncation=True, return_tensors='pt')
                mrs_only = tokenizer(batch[1], add_special_tokens=False, max_length=config.max_seq_length,
                                     padding=True, truncation=True, return_tensors='pt')

                input_ids = inputs['input_ids']
                input_mask = inputs['attention_mask']
                mr_mask = mrs_only['attention_mask']

                label_mask = torch.zeros_like(input_ids)
                label_mask[:, :mr_mask.shape[1]] = mr_mask
                label_mask[input_ids == tokenizer.pad_token_id] = 1

                label_ids = input_ids.masked_fill(label_mask, -100)

            input_tensor = input_ids.to(device)
            mask_tensor = input_mask.to(device)
            label_tensor = label_ids.to(device)

            # DEBUG
            # print()
            # print('>> ENCODED inputs:\n', input_ids)
            # print('>> DECODED inputs:\n', tokenizer.decode(input_ids[0]))
            # print()
            # print('>> ENCODED input masks:\n', input_mask)
            # print()
            # print('>> ENCODED labels:\n', label_ids)
            # print('>> DECODED labels:\n', tokenizer.decode(label_ids[0][label_ids[0] >= 0]))
            # print()
            # print('>> ENCODED decoder masks:\n', decoder_mask)
            # print()

            model.train()

            # Clear previously calculated gradients (must perform before a backward pass, unless using RNNs)
            model.zero_grad()

            if config.fp16:
                # Forward pass
                with torch.cuda.amp.autocast():
                    if is_enc_dec:
                        loss = model(input_tensor, attention_mask=mask_tensor,
                                     decoder_input_ids=decoder_input_tensor,
                                     # decoder_attention_mask=decoder_mask_tensor,
                                     labels=label_tensor, use_cache=False)[0]
                    else:
                        loss = model(input_tensor, attention_mask=mask_tensor, labels=label_tensor)[0]

                # Accumulate the training loss
                train_loss_sum += loss.item()

                # Backward pass
                scaler.scale(loss).backward()

                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)

                # Clip the norm of the gradients (in order to prevent the gradients from exploding)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                if is_enc_dec:
                    loss = model(input_tensor, attention_mask=mask_tensor,
                                 decoder_input_ids=decoder_input_tensor,
                                 # decoder_attention_mask=decoder_mask_tensor,
                                 labels=label_tensor, use_cache=False)[0]
                else:
                    loss = model(input_tensor, attention_mask=mask_tensor, labels=label_tensor)[0]

                # Accumulate the training loss
                train_loss_sum += loss.item()

                # Backward pass
                loss.backward()

                # Clip the norm of the gradients (in order to prevent the gradients from exploding)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()

            # Update the learning rate according to the defined schedule
            scheduler.step()

            steps_since_last_eval += 1

            if step in eval_steps:
                # Print stats
                avg_train_loss = train_loss_sum / steps_since_last_eval
                print()
                print('>> Training loss:  \t{0:.4f}'.format(avg_train_loss))
                print()
                train_loss_sum = 0.0
                steps_since_last_eval = 0

                # Validation
                scores = validate(config, valid_data_loader, tokenizer, model, is_enc_dec, device=device)
                scores_bleu = validate_bleu(config, valid_set_grouped, valid_grouped_data_loader, tokenizer, model,
                                            is_enc_dec, device=device)
                metrics = {**scores, **scores_bleu}

                print()
                print('>> Validation loss: {0:.4f}'.format(metrics.get('loss').item()))
                print('>> Validation PPL: {0:.4f}'.format(metrics.get('perplexity').item()))
                print('>> Validation BLEU: {0:.4f}'.format(metrics.get('bleu').item()))
                print('>> Validation BLEU (multi-ref): {0:.4f}'.format(metrics.get('bleu_multiref').item()))
                print()

                # Save a model checkpoint
                save_model(model, config.pretrained_model, epoch, step)

    model_dir = os.path.join('seq2seq', 'model', 'final')
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def validate(config, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    """Generates token ID sequences with teacher forcing, and calculates the loss and perplexity using references."""

    eval_loss_sum = 0.0

    model.eval()

    for batch in tqdm(data_loader, desc='Evaluating'):
        if is_enc_dec:
            # inputs = tokenizer.prepare_seq2seq_batch(batch[0], batch[1], max_length=config.max_seq_length,
            #                                          padding=True, truncation=True, return_tensors='pt')
            inputs = tokenizer(batch[0], add_special_tokens=True, max_length=config.max_seq_length,
                               padding=True, truncation=True, return_tensors='pt')
            targets = tokenizer(batch[1], add_special_tokens=True, max_length=config.max_seq_length,
                                padding=True, truncation=True, return_tensors='pt')

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            label_ids = targets['input_ids'].clone()
            label_ids[label_ids == tokenizer.pad_token_id] = -100

            if 'bart' in config.pretrained_model:
                decoder_input_ids = shift_tokens_right(targets['input_ids'], tokenizer.pad_token_id)
                # decoder_mask = targets['attention_mask']
                decoder_input_tensor = decoder_input_ids.to(device)
                # decoder_mask_tensor = decoder_mask.to(device)
            else:
                # Decoder input IDs and mask are inferred automatically from labels
                decoder_input_tensor = None
        else:
            inputs = tokenizer(batch[0], add_special_tokens=False, max_length=config.max_seq_length,
                               padding=True, truncation=True, return_tensors='pt')
            mrs_only = tokenizer(batch[1], add_special_tokens=False, max_length=config.max_seq_length,
                                 padding=True, truncation=True, return_tensors='pt')

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            mr_mask = mrs_only['attention_mask']

            label_mask = torch.zeros_like(input_ids)
            label_mask[:, :mr_mask.shape[1]] = mr_mask
            label_mask[input_ids == tokenizer.pad_token_id] = 1

            label_ids = input_ids.masked_fill(label_mask, -100)

        input_tensor = input_ids.to(device)
        mask_tensor = input_mask.to(device)
        label_tensor = label_ids.to(device)

        with torch.no_grad():
            if is_enc_dec:
                loss = model(input_tensor, attention_mask=mask_tensor,
                             decoder_input_ids=decoder_input_tensor,
                             # decoder_attention_mask=decoder_mask_tensor,
                             labels=label_tensor, use_cache=False)[0]
            else:
                loss = model(input_tensor, attention_mask=mask_tensor, labels=label_tensor)[0]

            # Accumulate the evaluation loss
            eval_loss_sum += loss.item()

    eval_loss = torch.tensor(eval_loss_sum / len(data_loader))
    perplexity = torch.exp(eval_loss)

    result = {
        'loss': eval_loss,
        'perplexity': perplexity
    }

    return result


def validate_bleu(config, dataset, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    """Generates decoded utterances without teacher forcing, and calculates their BLEU score using references."""

    generated_utterances = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=True,
                                               device=device)
    generated_utterances_flat = list(chain.from_iterable(generated_utterances))

    # DEBUG
    # print('>> PREDICTIONS decoded:')
    # print('\n'.join(generated_utterances[:50]))

    bleu = calculate_singleref_bleu(dataset, generated_utterances_flat)
    bleu_multiref = calculate_multiref_bleu(dataset, generated_utterances_flat)

    result = {
        'bleu': torch.tensor(bleu),
        'bleu_multiref': torch.tensor(bleu_multiref)
    }

    return result


def generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=False, device='cpu'):
    generated_sequences = []

    for batch in tqdm(data_loader, desc='Evaluating'):
        inputs = tokenizer(batch[0], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')

        # DEBUG
        # print('>> ENCODED inputs:', inputs['input_ids'])
        # print('>> ENCODED mask:', inputs['attention_mask'])

        if is_validation:
            outputs = model.generate(inputs['input_ids'].to(device),
                                     attention_mask=inputs['attention_mask'].to(device),
                                     max_length=config.max_seq_length)
        else:
            outputs = model.generate(inputs['input_ids'].to(device),
                                     attention_mask=inputs['attention_mask'].to(device),
                                     max_length=config.max_seq_length,
                                     num_beams=config.num_beams,
                                     early_stopping=config.early_stopping,
                                     no_repeat_ngram_size=config.no_repeat_ngram_size,
                                     do_sample=config.do_sample,
                                     top_p=config.top_p,
                                     top_k=config.top_k,
                                     temperature=config.temperature,
                                     repetition_penalty=config.repetition_penalty,
                                     length_penalty=config.length_penalty,
                                     num_return_sequences=config.num_return_sequences,
                                     # bos_token_id=tokenizer.bos_token_id,
                                     # decoder_start_token_id=tokenizer.eos_token_id,
                                     # pad_token_id=tokenizer.pad_token_id
                                     )

        generated_sequences.append(decode_model_outputs(outputs, tokenizer, is_enc_dec))

    return generated_sequences


def decode_model_outputs(sequences, tokenizer, is_enc_dec):
    outputs_decoded = []

    # TODO: generalize to batch_size > 1
    for i, seq in enumerate(sequences):
        if is_enc_dec:
            utt_decoded = tokenizer.decode(seq, skip_special_tokens=True)
        else:
            utt_beg_pos = np.where(seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
            utt_decoded = tokenizer.decode(seq[utt_beg_pos:], skip_special_tokens=True)

        # print('>> Sample #{}: {}'.format(i, utt_decoded))
        outputs_decoded.append(utt_decoded)

    return outputs_decoded


def calculate_singleref_bleu(dataset, predictions):
    """Calculates the corpus BLEU score with a single reference per generated utterance.

    Assumes the dataset to be grouped by MR, and to thus have a list of reference utterances for each MR. This method
    flattens the references and multiplies the generated predictions as necessary to match corresponding references.
    """
    references = dataset.get_utterances(lowercased=True)

    # Multiply generated utterances depending on the number of corresponding references, and then flatten references
    predictions_multiplied = list(chain.from_iterable(
        [pred] * len(ref_list) for pred, ref_list in zip(predictions, references)))
    references_flat = list(chain.from_iterable(references))

    return corpus_bleu(predictions_multiplied, [references_flat]).score


def calculate_multiref_bleu(dataset, predictions):
    """Calculates the corpus BLEU score with multiple references per generated utterance.

    Assumes the dataset to be grouped by MR, and to thus have a list of reference utterances for each MR. Assumes the
    generated utterances to have been produced from unique inputs, and hence to be a flat list. This method transposes
    the nested list of reference utterances to conform with the format sacreblue's corpus_bleu method expects.
    """
    references = dataset.get_utterances(lowercased=True)

    # Only works if the number of references is the same for each input
    # references_transposed = list(map(list, zip(*references)))

    # Transpose the reference utterances
    max_num_refs = max(len(ref_list) for ref_list in references)
    references_transposed = [[] for _ in range(max_num_refs)]
    for ref_list in references:
        idx = 0
        for ref in ref_list:
            references_transposed[idx].append(ref)
            idx += 1

        # Pad with the first reference
        for i in range(idx, max_num_refs):
            references_transposed[i].append(ref_list[0])

    return corpus_bleu(predictions, references_transposed).score


def test(config, dataset_class, device='cpu'):
    # Load model and corresponding tokenizer
    if 'gpt2' in config.pretrained_model:
        loading_function = load_pretrained_gpt2_model_and_tokenizer
        is_enc_dec = False
    elif 'bart' in config.pretrained_model:
        loading_function = load_pretrained_bart_model_and_tokenizer
        is_enc_dec = True
    elif 't5' in config.pretrained_model:
        loading_function = load_pretrained_t5_model_and_tokenizer
        is_enc_dec = True
    else:
        print('Error: model "{}" not supported'.format(config.pretrained_model))
        sys.exit()

    # Load model and corresponding tokenizer
    model, tokenizer = loading_function(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    load_model_checkpoint(model, config.pretrained_model, config.checkpoint_epoch, config.checkpoint_step)
    model = model.to(device)
    model.eval()

    # Load test data
    test_set = dataset_class(tokenizer, 'test', lowercase=True, convert_slot_names=config.convert_slot_names,
                             group_by_mr=True, separate_source_and_target=is_enc_dec)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Generate decoded utterances
    predictions = generate_and_decode(config, test_data_loader, tokenizer, model, is_enc_dec, device=device)

    # Make sure the output directory exists for the given dataset
    predictions_dir = os.path.join('seq2seq', 'predictions', test_set.name)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Prepare the metrics script command, and create the reference file for the given dataset
    eval_dir = os.path.join('seq2seq', 'eval')
    metrics_script = 'python ' + os.path.join(eval_dir, 'E2E', 'measure_scores.py')
    reference_file = os.path.join(eval_dir, 'test_references_{}.txt'.format(test_set.name))
    if not os.path.exists(reference_file):
        print('>> Generating a reference file for the "{}" test set.'.format(test_set.name))
        test_set.create_reference_file_for_testing()

    eval_configurations = []

    if config.semantic_reranking:
        predictions_reranked = rerank_beams(predictions, test_set.get_mrs_as_dicts())
        predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
        eval_configurations.append((predictions_reranked, True))

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    eval_configurations.insert(0, (predictions, False))

    for prediction_list, reranked in eval_configurations:
        file_name_root = compose_output_file_name(config, reranked=reranked)

        # Save generated utterances along with their corresponding MRs into a CSV file
        file_name = f'{file_name_root}.csv'
        df_predictions = pd.DataFrame({'mr': test_set.get_mrs(raw=True), 'utt': prediction_list})
        df_predictions.to_csv(os.path.join(predictions_dir, file_name), index=False, encoding='utf-8-sig')

        # Save generated utterances in a text file (for reference-based metric evaluation)
        file_name = f'{file_name_root}_utt_only.txt'
        predictions_file = os.path.join(predictions_dir, file_name)
        with open(predictions_file, 'w') as f_out:
            for prediction in prediction_list:
                f_out.write(prediction + '\n')

        # Run the metrics script provided by the E2E NLG Challenge
        os.system(metrics_script + ' ' + reference_file + ' ' + predictions_file)


def compose_output_file_name(config, reranked=False):
    if config.num_beams > 1:
        inference_method_suffix = '_beam_search_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.length_penalty)
    elif config.do_sample and config.top_p < 1.0:
        inference_method_suffix = '_nucleus_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_p)
    elif config.do_sample and config.top_k > 0:
        inference_method_suffix = '_top_k_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_k)
    else:
        inference_method_suffix = '_no_beam_search'

    file_name = 'epoch_{}_step_{}{}'.format(config.checkpoint_epoch, config.checkpoint_step, inference_method_suffix)

    return file_name


def generate_from_input(input_str, config, dataset_class, device='cpu'):
    # Load model and corresponding tokenizer
    model, tokenizer = load_pretrained_gpt2_model_and_tokenizer(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    load_model_checkpoint(model, config.pretrained_model, config.checkpoint_epoch, config.checkpoint_step)
    model = model.to(device)
    model.eval()

    input_ids = tokenizer(input_str)['input_ids']
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    outputs = model.generate(input_tensor,
                             max_length=config.max_seq_length,
                             num_beams=config.num_beams,
                             early_stopping=config.early_stopping,
                             no_repeat_ngram_size=config.no_repeat_ngram_size,
                             do_sample=config.do_sample,
                             top_p=config.top_p,
                             top_k=config.top_k,
                             temperature=config.temperature,
                             repetition_penalty=config.repetition_penalty,
                             length_penalty=config.length_penalty,
                             num_return_sequences=config.num_return_sequences,
                             bos_token_id=tokenizer.bos_token_id,
                             pad_token_id=tokenizer.pad_token_id)

    for i, output_seq in enumerate(outputs):
        utt_beg_pos = np.where(output_seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
        utt_decoded = tokenizer.decode(output_seq[utt_beg_pos:], skip_special_tokens=True)
        print('>> Sample #{}: {}'.format(i, utt_decoded))


def rerank_beams(beams, mrs, keep_n=None, keep_least_errors_only=False):
    """Reranks beams based on the slot error rate determined by the slot aligner. Keeps at most n best candidates.

    Note: Python's sort is guaranteed to be stable, i.e., when multiple records have the same key (e.g., slot error
    score), their original order (e.g., based on their beam score) is preserved.
    """
    beams_reranked = []

    for idx, mr in enumerate(tqdm(mrs, desc='Reranking')):
        beam_scored = []

        for utt in beams[idx]:
            # Calculate the slot error score
            score = score_alignment(utt, mr)
            beam_scored.append((utt, score))

        # Rerank utterances by slot error score (the higher the better)
        beam_scored.sort(key=lambda tup: tup[1], reverse=True)

        if keep_least_errors_only:
            # Filter only those utterances that have the least number of errors identified by the slot aligner
            beam_scored = [candidate for candidate in beam_scored if candidate[1] == beam_scored[0][1]]

        # Keep at most n candidates
        if keep_n is not None and len(beam_scored) > keep_n > 0:
            beam_scored = beam_scored[:keep_n]

        # DEBUG
        # if idx < 5:
        #     print('>> Scored beams:')
        #     print('\n'.join('{0} :: {1}'.format(utt[1], utt[0]) for utt in beam_scored))
        #     print()

        # Store the reranked beam (utterances only)
        beams_reranked.append([utt[0] for utt in beam_scored])

    return beams_reranked


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Training/test config name')
    parser.add_argument('-d', '--dataset', required=True, choices=['rest_e2e', 'rest_e2e_cleaned', 'video_game'],
                        help='Dataset name')
    parser.add_argument('-t', '--task', required=True, choices=['train', 'test'],
                        help='Task (train or test)')
    args = parser.parse_args()

    # Get the corresponding dataset class
    if args.dataset == 'rest_e2e':
        dataset_class = E2EDataset
    elif args.dataset == 'rest_e2e_cleaned':
        dataset_class = E2ECleanedDataset
    elif args.dataset == 'video_game':
        dataset_class = ViggoDataset
    else:
        print('Error: dataset "{}" not recognized'.format(args.dataset))
        sys.exit()

    # Validate the task name
    if args.task not in ['train', 'test', 'generate']:
        print('Error: task "{}" not recognized'.format(args.task))
        sys.exit()

    # Load the task configuration
    config = load_config(args.config, args.dataset, args.task, print_config=True)

    # Set the device to GPU if available, or CPU otherwise
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPUs available:', torch.cuda.device_count())
        print('CUDA version:', torch.version.cuda)
        print()
    else:
        device = 'cpu'

    # Run the corresponding task
    if args.task == 'train':
        save_training_config(config)
        train(TrainingConfig(config), dataset_class, device=device)
    elif args.task == 'test':
        test(TestConfig(config), dataset_class, device=device)
    elif args.task == 'generate':
        input_str = '<|name|> alimentum <|area|> city centre <|familyfriendly|> no <|begoftext|>'
        generate_from_input(input_str, TestConfig(config), dataset_class, device=device)


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    main()
