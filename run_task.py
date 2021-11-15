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

from data_loader import (
    E2EDataset, E2ECleanedDataset,
    MultiWOZDataset,
    ViggoDataset, ViggoWithE2EDataset, Viggo20PercentDataset, Viggo10PercentDataset, Viggo5PercentDataset,
    Viggo2PercentDataset, Viggo1PercentDataset)
from decoding import generate_and_decode
import eval_utils as eval_utils
import model_utils as model_utils
from task_config import TestConfig, TrainingConfig


torch.manual_seed(0)


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


def test(config, test_set, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    eval_configurations = []

    # Generate decoded utterances
    predictions, slot_errors = generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec,
                                                   bool_slots=test_set.bool_slots, device=device)

    if config.semantic_reranking or config.semantic_reranking_all:
        if config.semantic_reranking_all or not config.semantic_decoding:
            # Rerank generated beams based on semantic accuracy determined by the slot aligner
            predictions_reranked = eval_utils.rerank_beams(
                predictions, test_set.get_mrs(convert_slot_names=True), test_set.name
            )
            predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
            eval_configurations.append((predictions_reranked, True, None))

        if config.semantic_decoding:
            # Rerank generated beams based on semantic accuracy determined by attention tracking
            predictions_reranked, slot_errors_reranked = eval_utils.rerank_beams_attention_based(
                predictions, slot_errors
            )
            predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
            slot_errors_reranked = [slot_error_beam[0] for slot_error_beam in slot_errors_reranked]
            eval_configurations.append((predictions_reranked, True, slot_errors_reranked))

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    if slot_errors:
        slot_errors = [slot_error_beam[0] for slot_error_beam in slot_errors]
    eval_configurations.insert(0, (predictions, False, slot_errors))

    if slot_errors:
        eval_utils.save_slot_errors(config, test_set, eval_configurations)

    if test_set.name == 'multiwoz':
        # Evaluate the generated utterances on the BLEU metric with just single references
        bleu = eval_utils.calculate_singleref_bleu(test_set, predictions)
        print()
        print(f'BLEU (single-ref): {bleu:.4f}')
        print()

    # Run reference-based evaluation of the generated utterances
    return eval_utils.execute_e2e_evaluation_script(config, test_set, eval_configurations)


def batch_test(config, dataset_class, device='cpu'):
    test_scores = {'not_reranked': [], 'reranked': [], 'reranked_att': []}

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

    eval_utils.print_test_scores(test_scores, output_dir=os.path.join('predictions', test_set.name))


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
        'rest_e2e', 'rest_e2e_cleaned', 'multiwoz', 'video_game', 'video_game_with_rest_e2e', 'video_game_20_percent',
        'video_game_10_percent', 'video_game_5_percent', 'video_game_2_percent', 'video_game_1_percent'],
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
    elif args.dataset == 'video_game_20_percent':
        dataset_class = Viggo20PercentDataset
    elif args.dataset == 'video_game_10_percent':
        dataset_class = Viggo10PercentDataset
    elif args.dataset == 'video_game_5_percent':
        dataset_class = Viggo5PercentDataset
    elif args.dataset == 'video_game_2_percent':
        dataset_class = Viggo2PercentDataset
    elif args.dataset == 'video_game_1_percent':
        dataset_class = Viggo1PercentDataset
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
        # input_str = "recommend(name[F1 2014], genres[driving/racing, simulation, sport], platforms[PC])"
        input_str = "inform(name[Quantum Break], release_year[2016], rating[average], genres[adventure, shooter], player_perspective[third person])"

        generate_from_input(TestConfig(config), input_str, dataset_class, device=device)


if __name__ == '__main__':
    main()
