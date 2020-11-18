import argparse
import copy
from itertools import chain
import numpy as np
import os
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

    # Load training and validation data
    train_set = dataset_class(tokenizer, 'train', lowercase=True, convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec)
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_set = dataset_class(tokenizer, 'valid', lowercase=True, convert_slot_names=config.convert_slot_names,
                              separate_source_and_target=is_enc_dec)
    valid_data_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    valid_set_grouped = dataset_class(tokenizer, 'valid', lowercase=True, convert_slot_names=config.convert_slot_names,
                                      group_by_mr=True, no_target=True, separate_source_and_target=is_enc_dec)
    valid_grouped_data_loader = DataLoader(valid_set_grouped, batch_size=config.batch_size, shuffle=False, num_workers=0)

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
                    scores_bleu = validate_bleu(config, valid_set_grouped, valid_grouped_data_loader, tokenizer, model,
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
                model_utils.save_model(model, config.model_name, epoch, step)

        # DEBUG
        print()
        print('>> Longest source sequence:', longest_source_seq)
        print('>> Longest target sequence:', longest_target_seq)
        print()

    model_dir = os.path.join('seq2seq', 'model', 'final')
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

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

    generated_utterances = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=True,
                                               device=device)
    generated_utterances_flat = list(chain.from_iterable(generated_utterances))

    # DEBUG
    # print('>> PREDICTIONS decoded:')
    # print('\n'.join(generated_utterances[:50]))

    bleu = eval_utils.calculate_singleref_bleu(dataset, generated_utterances_flat)
    bleu_multiref = eval_utils.calculate_multiref_bleu(dataset, generated_utterances_flat)

    result = {
        'BLEU': torch.tensor(bleu),
        'BLEU (multi-ref)': torch.tensor(bleu_multiref)
    }

    return result


def generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=False, device='cpu'):
    generated_sequences = []

    if 'gpt2' in config.model_name:
        # Set the tokenizer to padding input sequences on the left side in order to enable batch inference
        tokenizer.padding_side = 'left'

    for batch in tqdm(data_loader, desc='Evaluating'):
        batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=False)

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

    return generated_sequences


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


def test(config, test_set, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    eval_configurations = []

    # Generate decoded utterances
    predictions = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, device=device)

    if config.semantic_reranking:
        # Rerank generated beams based on semantic accuracy
        predictions_reranked = eval_utils.rerank_beams(predictions, test_set.get_mrs_as_dicts())
        predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
        eval_configurations.append((predictions_reranked, True))

    if test_set.name == 'multiwoz':
        generated_utterances_flat = list(chain.from_iterable(predictions))
        bleu = eval_utils.calculate_singleref_bleu(test_set, generated_utterances_flat)
        print()
        print(f'BLEU (single-ref): {bleu:.4f}')
        print()

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    eval_configurations.insert(0, (predictions, False))

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

    # Load test data
    test_set = dataset_class(tokenizer, 'test', lowercase=True, convert_slot_names=config.convert_slot_names,
                             group_by_mr=True, no_target=True, separate_source_and_target=is_enc_dec)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    test_set_ppl = dataset_class(tokenizer, 'test', lowercase=True, convert_slot_names=config.convert_slot_names,
                                 separate_source_and_target=is_enc_dec)
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


def generate_from_input(input_str, config, dataset_class, device='cpu'):
    # Load model and corresponding tokenizer
    model, tokenizer = model_utils.load_gpt2_model_and_tokenizer(
        config.model_name,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    model_utils.load_model_checkpoint(model, config.model_name, config.checkpoint_epoch, config.checkpoint_step)
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
                             )

    for i, output_seq in enumerate(outputs):
        utt_beg_pos = np.where(output_seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
        utt_decoded = tokenizer.decode(output_seq[utt_beg_pos:], skip_special_tokens=True)
        print('>> Sample #{}: {}'.format(i, utt_decoded))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Training/test config name')
    parser.add_argument('-d', '--dataset', required=True, choices=[
        'rest_e2e', 'rest_e2e_cleaned', 'multiwoz', 'video_game', 'video_game_with_rest_e2e', 'video_game_20'],
                        help='Dataset name')
    parser.add_argument('-t', '--task', required=True, choices=['train', 'test'],
                        help='Task (train or test)')
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
        input_str = '<|name|> alimentum <|area|> city centre <|familyfriendly|> no <|begoftext|>'
        generate_from_input(input_str, TestConfig(config), dataset_class, device=device)


if __name__ == '__main__':
    main()
