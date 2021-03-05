import argparse
from collections import OrderedDict
import copy
from itertools import chain
import numpy as np
import os
import pickle
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from seq2seq.data_loader import (
    E2EDataset, E2ECleanedDataset,
    MultiWOZDataset,
    ViggoDataset, ViggoWithE2EDataset, Viggo20Dataset)
import seq2seq.eval_utils as eval_utils
import seq2seq.model_utils as model_utils
from seq2seq.task_config import TestConfig, TrainingConfig


def train(config, dataset_class, device='cpu'):
    train_loss_sum = 0.0
    steps_since_last_eval = 0
    best_checkpoints = {
        'loss': (-1, -1, np.inf),
        'perplexity': (-1, -1, np.inf),
        'BLEU': (-1, -1, 0.0),
        'BLEU (multi-ref)': (-1, -1, 0.0),
    }

    # DEBUG
    longest_source_seq = 0
    longest_target_seq = 0

    # Load model and the corresponding tokenizer
    special_tokens = dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names)
    model, tokenizer = model_utils.load_model_and_tokenizer(config, special_tokens=special_tokens)
    is_enc_dec = model.config.is_encoder_decoder
    model = model.to(device)

    prepare_token_types = True if 'gpt2' in config.model_name and config.use_token_type_ids else False
    group_by_mr = False if dataset_class.name in {'multiwoz'} else True

    # Load training and validation data
    train_set = dataset_class(tokenizer,
                              partition='train',
                              lowercase=config.lowercase,
                              convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec,
                              prepare_token_types=prepare_token_types,
                              num_slot_permutations=config.num_slot_permutations)
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_set = dataset_class(tokenizer,
                              partition='valid',
                              lowercase=config.lowercase,
                              convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec,
                              sort_by_length=True,
                              prepare_token_types=prepare_token_types)
    valid_data_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    valid_set_bleu = dataset_class(tokenizer,
                                   partition='valid',
                                   lowercase=config.lowercase,
                                   convert_slot_names=config.convert_slot_names,
                                   group_by_mr=group_by_mr,
                                   no_target=True,
                                   separate_source_and_target=is_enc_dec,
                                   sort_by_length=True,
                                   prepare_token_types=prepare_token_types)
    valid_bleu_data_loader = DataLoader(valid_set_bleu, batch_size=config.eval_batch_size, shuffle=False, num_workers=0)

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

    # Save the training config in the model output folder
    model_utils.save_training_config(config)

    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        print()
        print(' ******************* ')
        print('**   EPOCH {:>2}/{:<2}   **'.format(epoch, config.num_epochs))
        print(' ******************* ')
        print()

        for step, batch in enumerate(tqdm(train_data_loader, desc='Step'), start=1):
            batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=True)

            # DEBUG
            if batch['input_ids'].size(1) > longest_source_seq:
                longest_source_seq = batch['input_ids'].size(1)
            if batch['labels'].size(1) > longest_target_seq:
                longest_target_seq = batch['labels'].size(1)

            input_tensor = batch['input_ids'].to(device)
            mask_tensor = batch['attention_mask'].to(device)
            label_tensor = batch['labels'].to(device)

            model_specific_args = {}
            if batch.get('token_type_ids') is not None:
                model_specific_args['token_type_ids'] = batch['token_type_ids'].to(device)
            if batch.get('decoder_input_ids') is not None:
                model_specific_args['decoder_input_ids'] = batch['decoder_input_ids'].to(device)

            # Set model to training mode
            model.train()

            # Clear previously calculated gradients (must perform before a backward pass, unless using RNNs)
            model.zero_grad()

            if config.fp16:
                # Forward pass
                with torch.cuda.amp.autocast():
                    loss = model(input_tensor,
                                 attention_mask=mask_tensor,
                                 labels=label_tensor,
                                 use_cache=False,
                                 **model_specific_args)[0]

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
                loss = model(input_tensor,
                             attention_mask=mask_tensor,
                             labels=label_tensor,
                             use_cache=False,
                             **model_specific_args)[0]

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

                if epoch % config.eval_every_n_epochs == 0:
                    # Validation
                    scores = validate(config, valid_data_loader, tokenizer, model, is_enc_dec, device=device)
                    scores_bleu = validate_bleu(config, valid_set_bleu, valid_bleu_data_loader, tokenizer, model,
                                                is_enc_dec, device=device)
                    metrics = {**scores, **scores_bleu}

                    # Print validation metrics, and keep track of the best checkpoints based on each metric
                    print()
                    for metric in ['loss', 'perplexity', 'BLEU', 'BLEU (multi-ref)']:
                        metric_val = metrics.get(metric).item()
                        print('>> Validation {}: {:.4f}'.format(metric, metric_val))

                        if metric in ['loss', 'perplexity']:
                            if metric_val < best_checkpoints[metric][2]:
                                best_checkpoints[metric] = (epoch, step, metric_val)
                        else:
                            if metric_val > best_checkpoints[metric][2]:
                                best_checkpoints[metric] = (epoch, step, metric_val)
                    print()

                # Save a model checkpoint
                model_utils.save_model(model, tokenizer, epoch, step)

        # DEBUG
        print()
        print('>> Longest source sequence:', longest_source_seq)
        print('>> Longest target sequence:', longest_target_seq)
        print()

    # Print an overview of the best checkpoints by metric
    eval_utils.print_best_checkpoints(best_checkpoints)


def validate(config, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    """Generates token ID sequences with teacher forcing, and calculates the loss and perplexity using references."""

    eval_loss_sum = 0.0

    # Set model to evaluation mode
    model.eval()

    for batch in tqdm(data_loader, desc='Evaluating'):
        batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=True)

        input_tensor = batch['input_ids'].to(device)
        mask_tensor = batch['attention_mask'].to(device)
        label_tensor = batch['labels'].to(device)

        model_specific_args = {}
        if batch.get('token_type_ids') is not None:
            model_specific_args['token_type_ids'] = batch['token_type_ids'].to(device)
        if batch.get('decoder_input_ids') is not None:
            model_specific_args['decoder_input_ids'] = batch['decoder_input_ids'].to(device)

        with torch.no_grad():
            # Forward pass
            loss = model(input_tensor,
                         attention_mask=mask_tensor,
                         labels=label_tensor,
                         use_cache=False,
                         **model_specific_args)[0]

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

    generated_utterances, _ = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=True,
                                                  device=device)
    generated_utterances_flat = list(chain.from_iterable(generated_utterances))

    # DEBUG
    # print('>> PREDICTIONS decoded:')
    # print('\n'.join(generated_utterances[:50]))

    bleu = eval_utils.calculate_singleref_bleu(dataset, generated_utterances_flat)
    # TODO: add a flag to the Dataset class indicating whether the dataset has multiple references or not
    if dataset.name not in {'multiwoz'}:
        bleu_multiref = eval_utils.calculate_multiref_bleu(dataset, generated_utterances_flat)
    else:
        bleu_multiref = bleu

    result = {
        'BLEU': torch.tensor(bleu),
        'BLEU (multi-ref)': torch.tensor(bleu_multiref)
    }

    return result


def generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=False, bool_slots=None,
                        device='cpu'):
    generated_sequences = []
    slot_errors = []

    if 'gpt2' in config.model_name:
        # Set the tokenizer to padding input sequences on the left side in order to enable batch inference
        tokenizer.padding_side = 'left'

    for batch in tqdm(data_loader, desc='Generating'):
        batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=False,
                                          bool_slots=bool_slots)

        input_tensor = batch['input_ids'].to(device)
        mask_tensor = batch['attention_mask'].to(device)

        model_specific_args = {}
        if batch.get('token_type_ids') is not None:
            model_specific_args['token_type_ids'] = batch['token_type_ids'].to(device)
            if hasattr(config, 'num_return_sequences'):
                model_specific_args['expand_token_type_size'] = config.num_return_sequences

        # DEBUG
        # print()
        # print('>> ENCODED inputs:', inputs['input_ids'])
        # print('>> DECODED inputs:\n', tokenizer.decode(inputs['input_ids'][0]))
        # print()
        # print('>> ENCODED input mask:', inputs['attention_mask'])
        # print()

        if is_validation:
            num_seqs_per_input = 1
            outputs = model.generate(input_tensor,
                                     attention_mask=mask_tensor,
                                     min_length=1,
                                     max_length=config.max_seq_length,
                                     **model_specific_args)
        else:
            if config.semantic_decoding:
                num_seqs_per_input = 1
                outputs, attn_weights, slot_error_list = semantic_decoding(input_tensor,
                                                                           batch['slot_spans'],
                                                                           tokenizer,
                                                                           model,
                                                                           max_length=config.max_seq_length,
                                                                           device=device)

                slot_errors.extend(slot_error_list)

                # # Save the input and output sequences (as lists of tokens) along with the attention weights
                # attn_weights['input_tokens'] = [tokenizer.decode(input_id, skip_special_tokens=False)
                #                                 for input_id in batch['input_ids'][0]]
                # attn_weights['output_tokens'] = [tokenizer.decode(output_id, skip_special_tokens=False)
                #                                  for output_id in outputs[0][1:]]
                #
                # # Export attention weights for visualization
                # with open(os.path.join('seq2seq', 'attention', 'attention_weights.pkl'), 'wb') as f_attn:
                #     pickle.dump(attn_weights, f_attn)
            else:
                num_seqs_per_input = config.num_return_sequences
                outputs = model.generate(input_tensor,
                                         attention_mask=mask_tensor,
                                         min_length=1,
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
                                         **model_specific_args)

        generated_sequences.extend(decode_model_outputs(outputs, num_seqs_per_input, tokenizer, is_enc_dec))

    if 'gpt2' in config.model_name:
        # Set the tokenizer back to padding input sequences on the right
        tokenizer.padding_side = 'right'

    return generated_sequences, slot_errors


def decode_model_outputs(sequences, num_seqs_per_input, tokenizer, is_enc_dec):
    outputs_decoded = []

    for beam_idx in range(0, len(sequences), num_seqs_per_input):
        beam_sequences = sequences[beam_idx:beam_idx + num_seqs_per_input]
        beam_decoded = []

        for seq in beam_sequences:
            if is_enc_dec:
                utt_decoded = tokenizer.decode(seq, skip_special_tokens=True)
            else:
                utt_beg_pos = np.where(seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
                utt_decoded = tokenizer.decode(seq[utt_beg_pos:], skip_special_tokens=True)

            beam_decoded.append(utt_decoded)

        outputs_decoded.append(beam_decoded)

    return outputs_decoded


def semantic_decoding(input_ids, slot_spans, tokenizer, model, max_length=128, device='cpu'):
    """Performs a semantically guided inference from a structured MR input.

    Note: currently, this performs a simple greedy decoding.
    """
    outputs = None
    attention_weights = {}

    # Initialize the decoder's input sequence with the corresponding token
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long).to(device)

    # Run the input sequence through the encoder and get the encoded input sequence
    encoder = model.get_encoder()
    encoded_sequence = encoder(input_ids, output_attentions=True)

    # Save the encoder's self-attention weights
    attention_weights['enc_attn_weights'] = [weight_tensor.squeeze().tolist() for weight_tensor in encoded_sequence.attentions]

    for step in range(max_length):
        # Reuse the encoded inputs, and pass the sequence generated so far as inputs to the decoder
        outputs = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids,
                        output_attentions=True, return_dict=True)

        logits = outputs.logits

        next_decoder_input_ids = select_next_token(logits, outputs.cross_attentions, slot_spans, tokenizer.eos_token_id)

        # Select the token with the highest probability as the next generated token (~ greedy decoding)
        next_decoder_input_ids = torch.argmax(logits[:, -1, :], axis=-1)

        # Append the current output token's ID to the sequence generated so far
        decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids.unsqueeze(-1)], axis=-1)

        # DEBUG
        # for i in range(len(outputs.cross_attentions)):
        #     print(outputs.cross_attentions[i].size())
        # print()

        # Terminate as soon as the decoder generates the EOS token
        if next_decoder_input_ids.item() == tokenizer.eos_token_id:
            break

    if outputs:
        # Save the decoder's self- and cross-attention weights; shape = (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        attention_weights['dec_attn_weights'] = [weight_tensor.squeeze().tolist()
                                                 for weight_tensor in outputs.decoder_attentions]
        attention_weights['cross_attn_weights'] = [weight_tensor.squeeze().tolist()
                                                   for weight_tensor in outputs.cross_attentions]

    # DEBUG
    # print('>> Slot mentions:')
    # print(slot_spans)
    # print()

    slot_errors = evaluate_slot_mentions(slot_spans)

    # DEBUG
    # print('>> Slot errors:')
    # print(slot_errors)
    # print()

    return decoder_input_ids, attention_weights, slot_errors


def evaluate_slot_mentions(slot_mentions_batch):
    slot_errors_batch = []

    for slot_mentions in slot_mentions_batch:
        slot_errors = []

        for slot in slot_mentions:
            # If any of the slot's values were not mentioned, consider the slot mention erroneous
            if not all(slot['mentioned']):
                slot_errors.append(slot['name'])

        slot_errors_batch.append(slot_errors)

    return slot_errors_batch


def select_next_token(logits, attn_weights, slot_spans, eos_token_id):
    # DEBUG
    # print('>> logits.size():', logits.size())
    # print('>> attn_weights[0].shape:', attn_weights[0].shape)
    # print()

    # Convert from a tuple to an array, and remove the batch dimension
    attn_weights = np.stack([layer.detach().cpu().numpy().squeeze(axis=0) for layer in attn_weights])
    # Ignore weights from other than the most recent step in the sequence
    attn_weights = attn_weights[:, :, -1:, :]

    # DEBUG
    # print('>> attn_weights.shape (current time step only):', attn_weights.shape)
    # print()

    # Extract the attention weights from the 1st decoder layer only, and aggregate them across heads
    attn_weights_first_layer = preprocess_attn_weights(
        attn_weights[0:1, :, :, :], head_agg_mode='max', layer_agg_mode=None)
    attn_weights_first_layer = binarize_weights(
        attn_weights_first_layer, threshold=0.5, keep_max_only=True).squeeze(axis=(1, 2))
    attn_idxs = np.where(attn_weights_first_layer == 1)

    # Update slot mentions with a high confidence
    update_slot_mentions(slot_spans, attn_idxs, confidence=True)

    if torch.argmax(logits[:, -1, :], axis=-1).item() == eos_token_id:
        # Remove slot mentions if they have a high attention weight associated with the EOS token
        attn_weights_agg = preprocess_attn_weights(attn_weights, head_agg_mode='max', layer_agg_mode='avg')
        attn_weights_agg = binarize_weights(attn_weights_agg, threshold=0.1).squeeze(axis=(1, 2))
        attn_idxs = np.where(attn_weights_agg == 1)

        remove_slot_mentions(slot_spans, attn_idxs)
    else:
        # Aggregate the attention weights across both the heads and the layers
        attn_weights_agg = preprocess_attn_weights(attn_weights, head_agg_mode='max', layer_agg_mode='avg')
        attn_weights_agg = binarize_weights(attn_weights_agg, threshold=0.3, keep_max_only=True).squeeze(axis=(1, 2))
        attn_idxs = np.where(attn_weights_agg == 1)

        # DEBUG
        # print('>> attn_weights_agg.shape (after binarizing):', attn_weights_agg.shape)
        # print('>> attn_idxs:', attn_idxs)
        # print()

        # Update slot mentions with a low confidence
        update_slot_mentions(slot_spans, attn_idxs, confidence=False)


def update_slot_mentions(slot_mention_batch, attn_idxs, confidence=False):
    for batch_idx, attn_idx in zip(attn_idxs[0], attn_idxs[1]):
        for slot in slot_mention_batch[batch_idx]:
            if all(slot['mentioned']) and all(slot['confidence']):
                continue

            attn_weight_matched = False

            if 'value_span' in slot and not slot['is_boolean']:
                for elem_idx, value_elem_span in enumerate(slot['value_span']):
                    # TODO: optimize by breaking out of the loop if attn_idx is less than the position of the 1st element or greater than the position of the last element
                    if value_elem_span[0] <= attn_idx <= value_elem_span[1]:
                        slot['mentioned'][elem_idx] = True
                        if not slot['confidence'][elem_idx]:
                            slot['confidence'][elem_idx] = confidence
                        attn_weight_matched = True
                        break
            else:
                # For Boolean slots and slots without a value, match the slot's name
                if slot['name_span'][0] <= attn_idx <= slot['name_span'][1]:
                    slot['mentioned'][0] = True
                    if not slot['confidence'][0]:
                        slot['confidence'][0] = confidence
                    attn_weight_matched = True

            if attn_weight_matched:
                break


def remove_slot_mentions(slot_mention_batch, attn_idxs):
    for batch_idx, attn_idx in zip(attn_idxs[0], attn_idxs[1]):
        for slot in slot_mention_batch[batch_idx]:
            if not any(slot['mentioned']):
                continue

            attn_weight_matched = False

            if 'value_span' in slot and not slot['is_boolean']:
                for elem_idx, value_elem_span in enumerate(slot['value_span']):
                    # TODO: optimize by breaking out of the loop if attn_idx is less than the position of the 1st element or greater than the position of the last element
                    if value_elem_span[0] <= attn_idx <= value_elem_span[1]:
                        if not slot['confidence'][elem_idx]:
                            slot['mentioned'][elem_idx] = False
                        attn_weight_matched = True
                        break
            else:
                # For Boolean slots and slots without a value, match the slot's name
                if slot['name_span'][0] <= attn_idx <= slot['name_span'][1]:
                    if not slot['confidence'][0]:
                        slot['mentioned'][0] = False
                    attn_weight_matched = True

            if attn_weight_matched:
                break


def preprocess_attn_weights(attn_weights, head_agg_mode=None, layer_agg_mode=None, threshold=0.0):
    if head_agg_mode:
        # num_heads = attn_weights.shape[1]
        attn_weights = aggregate_across_heads(attn_weights, mode=head_agg_mode)

    if layer_agg_mode:
        # num_layers = attn_weights.shape[0]
        # middle_layer_idx = num_layers // 2

        attn_weights = aggregate_across_layers(attn_weights, mode=layer_agg_mode)
        # attn_weights = aggregate_across_layers(attn_weights[0:middle_layer_idx, :, :, :], mode=layer_agg_mode)
        # for layer_idx in range(1, middle_layer_idx + 1):
        #     attn_weights_aggr = aggregate_across_layers(attn_weights[0:layer_idx, :, :, :], mode=layer_agg_mode)
        #     max_weights = attn_weights_aggr.max(axis=-1)[:, :, :, np.newaxis]
        #     attn_weights_aggr[np.nonzero(attn_weights_aggr < threshold)] = 0
        #     if (attn_weights_aggr == max_weights).any() or layer_idx == middle_layer_idx:
        #         attn_weights = attn_weights_aggr
        #         break

    return attn_weights


def aggregate_across_heads(attn_weights, mode='max'):
    """Sums weights across all heads, and normalizes the weights by these sums."""
    if mode == 'max':
        head_maxs = attn_weights.max(axis=1)
    elif mode == 'sum':
        head_maxs = attn_weights.sum(axis=1)
    elif mode == 'avg':
        head_maxs = attn_weights.mean(axis=1)
    else:
        raise ValueError(f'Aggregation mode "{mode}" unrecognized')

    return head_maxs[:, np.newaxis, :, :]


def aggregate_across_layers(attn_weights, mode='max'):
    """Sums weights across all layers, and normalizes the weights by these sums."""
    if mode == 'max':
        layer_sums = np.max(attn_weights, axis=0)
    elif mode == 'sum':
        layer_sums = np.sum(attn_weights, axis=0)
    elif mode == 'avg':
        layer_sums = np.mean(attn_weights, axis=0)
    else:
        raise ValueError(f'Aggregation mode "{mode}" unrecognized')

    return layer_sums[np.newaxis, :, :, :]


def binarize_weights(attn_weights, threshold=0.0, keep_max_only=False):
    if keep_max_only:
        max_weights = attn_weights.max(axis=-1)[:, :, :, np.newaxis]

        attn_weights[np.nonzero(attn_weights < threshold)] = 0
        attn_weights = (attn_weights == max_weights).astype(int)
    else:
        attn_weights[np.nonzero(attn_weights < threshold)] = 0
        attn_weights[np.nonzero(attn_weights >= threshold)] = 1

    return attn_weights


def test(config, test_set, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    eval_configurations = []

    # Generate decoded utterances
    predictions, slot_errors = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec,
                                                   bool_slots=test_set.bool_slots, device=device)

    if config.semantic_reranking:
        # Rerank generated beams based on semantic accuracy
        predictions_reranked = eval_utils.rerank_beams(
            predictions, test_set.get_mrs(convert_slot_names=True), test_set.name)
        predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
        eval_configurations.append((predictions_reranked, True))

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    eval_configurations.insert(0, (predictions, False))

    if slot_errors:
        # slot_errors = [slot_error_beam[0] for slot_error_beam in slot_errors]
        eval_utils.save_slot_errors(config, test_set, eval_configurations, slot_errors)

    if test_set.name == 'multiwoz':
        # Evaluate the generated utterances on the BLEU metric with just single references
        bleu = eval_utils.calculate_singleref_bleu(test_set, predictions)
        print()
        print(f'BLEU (single-ref): {bleu:.4f}')
        print()

    # Run reference-based evaluation of the generated utterances
    return eval_utils.execute_e2e_evaluation_script(config, test_set, eval_configurations)


def batch_test(config, dataset_class, device='cpu'):
    test_scores = {'not_reranked': [], 'reranked': []}

    # Load model and the corresponding tokenizer
    special_tokens = dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names)
    model, tokenizer = model_utils.load_model_and_tokenizer(config, special_tokens=special_tokens)
    is_enc_dec = model.config.is_encoder_decoder
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    prepare_token_types = True if 'gpt2' in config.model_name and config.use_token_type_ids else False
    group_by_mr = False if dataset_class.name in {'multiwoz'} else True

    # Load test data
    test_set = dataset_class(tokenizer,
                             partition='test',
                             lowercase=config.lowercase,
                             convert_slot_names=config.convert_slot_names,
                             group_by_mr=group_by_mr,
                             no_target=True,
                             separate_source_and_target=is_enc_dec,
                             prepare_token_types=prepare_token_types)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    test_set_ppl = dataset_class(tokenizer,
                                 partition='test',
                                 lowercase=config.lowercase,
                                 convert_slot_names=config.convert_slot_names,
                                 separate_source_and_target=is_enc_dec,
                                 prepare_token_types=prepare_token_types)
    test_data_loader_ppl = DataLoader(test_set_ppl, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Evaluate the model's perplexity
    scores = validate(config, test_data_loader_ppl, tokenizer, model, is_enc_dec, device=device)
    print()
    print('>> Test perplexity: {:.4f}'.format(scores.get('perplexity').item()))
    print()

    if isinstance(config.length_penalty, list):
        # Batch test with different length penalty values
        param_values = copy.deepcopy(config.length_penalty)
        for val in param_values:
            # Update the parameter value in the configuration and re-run testing
            config.length_penalty = val
            scores = test(config, test_set, test_data_loader, tokenizer, model, is_enc_dec, device=device)
            eval_utils.update_test_scores(test_scores, scores)
    elif isinstance(config.top_p, list):
        # Batch test with different p values for nucleus sampling
        param_values = copy.deepcopy(config.top_p)
        for val in param_values:
            # Update the parameter value in the configuration and re-run testing
            config.top_p = val
            scores = test(config, test_set, test_data_loader, tokenizer, model, is_enc_dec, device=device)
            eval_utils.update_test_scores(test_scores, scores)
    else:
        # Test with a single configuration
        scores = test(config, test_set, test_data_loader, tokenizer, model, is_enc_dec, device=device)
        eval_utils.update_test_scores(test_scores, scores)

    eval_utils.print_test_scores(test_scores, output_dir=os.path.join('seq2seq', 'predictions', test_set.name))


def generate_from_input(config, input_str, dataset_class, device='cpu'):
    # Load model and the corresponding tokenizer
    special_tokens = dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names)
    model, tokenizer = model_utils.load_model_and_tokenizer(config, special_tokens=special_tokens)
    is_enc_dec = model.config.is_encoder_decoder
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Create a data loader from the input string
    test_set = dataset_class(tokenizer,
                             input_str=input_str,
                             partition='test',
                             lowercase=config.lowercase,
                             no_target=True,
                             separate_source_and_target=is_enc_dec)
    test_data_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # Generate decoded utterances
    utterances, slot_errors = generate_and_decode(config, test_data_loader, tokenizer, model, is_enc_dec,
                                                  bool_slots=test_set.bool_slots, device=device)

    print('>> Generated utterance(s):')
    print('\n'.join(utterances[0]))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Training/test config name')
    parser.add_argument('-d', '--dataset', required=True, choices=[
        'rest_e2e', 'rest_e2e_cleaned', 'multiwoz', 'video_game', 'video_game_with_rest_e2e', 'video_game_20'],
                        help='Dataset name')
    parser.add_argument('-t', '--task', required=True, choices=['train', 'test', 'generate'],
                        help='Task (train, test, or generate)')
    args = parser.parse_args()

    # Get the corresponding dataset class
    if args.dataset == 'rest_e2e':
        dataset_class = E2EDataset
    elif args.dataset == 'rest_e2e_cleaned':
        dataset_class = E2ECleanedDataset
    elif args.dataset == 'multiwoz':
        dataset_class = MultiWOZDataset
    elif args.dataset == 'video_game':
        dataset_class = ViggoDataset
    elif args.dataset == 'video_game_with_rest_e2e':
        dataset_class = ViggoWithE2EDataset
        args.dataset = 'video_game'
    elif args.dataset == 'video_game_20':
        dataset_class = Viggo20Dataset
    else:
        print('Error: dataset "{}" not recognized'.format(args.dataset))
        sys.exit()

    # Validate the task name
    if args.task not in ['train', 'test', 'generate']:
        print('Error: task "{}" not recognized'.format(args.task))
        sys.exit()

    # Load the task configuration
    config = model_utils.load_config(args.config, args.dataset, args.task, print_config=True)

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
        train(TrainingConfig(config), dataset_class, device=device)
    elif args.task == 'test':
        batch_test(TestConfig(config), dataset_class, device=device)
    elif args.task == 'generate':
        # Restaurants (E2E)
        # input_str = "name[The Cricketers], eatType[restaurant], food[English], priceRange[high], customer rating[average], area[riverside], familyFriendly[no], near[Café Rouge]"
        # input_str = "name[The Phoenix], eatType[restaurant], food[Indian], priceRange[£20-25], customer rating[high], area[riverside], familyFriendly[yes], near[Crowne Plaza Hotel]"
        # input_str = "name[The Punter], eatType[restaurant], food[Italian], priceRange[cheap], customer rating[average], area[riverside], familyFriendly[yes], near[Rainbow Vegetarian Café]"
        # input_str = "name[The Wrestlers], eatType[pub], food[Japanese], priceRange[£20-25], area[riverside], familyFriendly[yes], near[Raja Indian Cuisine]"

        # Video games (ViGGO)
        # input_str = "inform(name[Tomb Raider: The Last Revelation], release_year[1999], esrb[T (for Teen)], genres[action-adventure, puzzle, shooter], platforms[PlayStation, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[yes])"
        # input_str = "inform(name[Assassin's Creed Chronicles: India], release_year[2016], genres[action-adventure, platformer], player_perspective[side view], has_multiplayer[no])"
        # input_str = "recommend(name[Crysis], has_multiplayer[yes], platforms[Xbox])"
        # input_str = "request_explanation(esrb[E (for Everyone)], rating[good], genres[adventure, platformer, puzzle])"
        # input_str = "verify_attribute(name[Uncharted 4: A Thief's End], esrb[T (for Teen)], rating[excellent])"
        # input_str = "request_explanation(rating[poor], genres[vehicular combat], player_perspective[third person])"
        # input_str = "inform(name[Tom Clancy's The Division], esrb[M (for Mature)], rating[average], genres[role-playing, shooter, tactical], player_perspective[third person], has_multiplayer[yes], platforms[PlayStation, Xbox, PC], available_on_steam[yes])"
        # input_str = "give_opinion(name[Mirror's Edge Catalyst], rating[poor], available_on_steam[no])"
        # input_str = "recommend(name[Madden NFL 15], genres[simulation, sport])"
        # input_str = "inform(name[World of Warcraft], release_year[2004], developer[Blizzard Entertainment], genres[adventure, MMORPG])"
        # input_str = "request(genres[driving/racing, simulation, sport], specifier[exciting])"
        # input_str = "inform(name[F1 2014], release_year[2014], rating[average], genres[driving/racing, simulation, sport])"
        # input_str = "suggest(name[The Sims], platforms[PC], available_on_steam[no])"
        # input_str = "request_attribute(developer[])"
        input_str = "recommend(name[F1 2014], genres[driving/racing, simulation, sport], platforms[PC])"

        generate_from_input(TestConfig(config), input_str, dataset_class, device=device)


if __name__ == '__main__':
    main()
