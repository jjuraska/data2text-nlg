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

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, ViggoDataset
from seq2seq.task_config import TestConfig, TrainingConfig
import seq2seq.eval_utils as eval_utils
import seq2seq.model_utils as model_utils


def train(config, dataset_class, device='cpu'):
    train_loss_sum = 0.0
    steps_since_last_eval = 0

    # Load model and the corresponding tokenizer
    special_tokens = dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names)
    model, tokenizer, is_enc_dec = model_utils.load_model_and_tokenizer(config, special_tokens=special_tokens)
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
        print(' ******************* ')
        print('**   EPOCH {:>2}/{:<2}   **'.format(epoch, config.num_epochs))
        print(' ******************* ')
        print()

        for step, batch in enumerate(tqdm(train_data_loader, desc='Step'), start=1):
            batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, device=device)

            input_tensor = batch['input_ids'].to(device)
            mask_tensor = batch['attention_mask'].to(device)
            label_tensor = batch['labels'].to(device)
            if batch.get('decoder_input_ids') is not None:
                decoder_input_tensor = batch['decoder_input_ids'].to(device)
            else:
                decoder_input_tensor = None

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

                if epoch % config.eval_every_n_epochs == 0:
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
                model_utils.save_model(model, config.model_name, epoch, step)

    model_dir = os.path.join('seq2seq', 'model', 'final')
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def validate(config, data_loader, tokenizer, model, is_enc_dec, device='cpu'):
    """Generates token ID sequences with teacher forcing, and calculates the loss and perplexity using references."""

    eval_loss_sum = 0.0

    model.eval()

    for batch in tqdm(data_loader, desc='Evaluating'):
        batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, device=device)

        input_tensor = batch['input_ids'].to(device)
        mask_tensor = batch['attention_mask'].to(device)
        label_tensor = batch['labels'].to(device)
        if batch.get('decoder_input_ids') is not None:
            decoder_input_tensor = batch['decoder_input_ids'].to(device)
        else:
            decoder_input_tensor = None

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

    bleu = eval_utils.calculate_singleref_bleu(dataset, generated_utterances_flat)
    bleu_multiref = eval_utils.calculate_multiref_bleu(dataset, generated_utterances_flat)

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
                                     max_length=config.max_seq_length,
                                     # bos_token_id=tokenizer.bos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     )
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
                                     pad_token_id=tokenizer.pad_token_id,
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


def test(config, dataset_class, device='cpu'):
    eval_configurations = []

    # Load model and the corresponding tokenizer
    special_tokens = dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names)
    model, tokenizer, is_enc_dec = model_utils.load_model_and_tokenizer(config, special_tokens=special_tokens)
    model = model.to(device)

    model.eval()

    # Load test data
    test_set = dataset_class(tokenizer, 'test', lowercase=True, convert_slot_names=config.convert_slot_names,
                             group_by_mr=True, no_target=True, separate_source_and_target=is_enc_dec)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Generate decoded utterances
    predictions = generate_and_decode(config, test_data_loader, tokenizer, model, is_enc_dec, device=device)

    if config.semantic_reranking:
        # Rerank generated beams based on semantic accuracy
        predictions_reranked = eval_utils.rerank_beams(predictions, test_set.get_mrs_as_dicts())
        predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
        eval_configurations.append((predictions_reranked, True))

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    eval_configurations.insert(0, (predictions, False))

    # Run reference-based evaluation of the generated utterances
    eval_utils.execute_e2e_evaluation_script(config, test_set, eval_configurations)


def generate_from_input(input_str, config, dataset_class, device='cpu'):
    # Load model and corresponding tokenizer
    model, tokenizer = model_utils.load_pretrained_gpt2_model_and_tokenizer(
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
                             bos_token_id=tokenizer.bos_token_id,
                             pad_token_id=tokenizer.pad_token_id)

    for i, output_seq in enumerate(outputs):
        utt_beg_pos = np.where(output_seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
        utt_decoded = tokenizer.decode(output_seq[utt_beg_pos:], skip_special_tokens=True)
        print('>> Sample #{}: {}'.format(i, utt_decoded))


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
        model_utils.save_training_config(config)
        train(TrainingConfig(config), dataset_class, device=device)
    elif args.task == 'test':
        test_config = TestConfig(config)
        if isinstance(test_config.length_penalty, list):
            # Batch test with different length penalty values
            param_values = copy.deepcopy(test_config.length_penalty)
            for val in param_values:
                # Update the parameter value in the configuration and re-run testing
                test_config.length_penalty = val
                test(test_config, dataset_class, device=device)
        elif isinstance(test_config.top_p, list):
            # Batch test with different p values for nucleus sampling
            param_values = copy.deepcopy(test_config.top_p)
            for val in param_values:
                # Update the parameter value in the configuration and re-run testing
                test_config.top_p = val
                test(test_config, dataset_class, device=device)
        else:
            # Test with a single configuration
            test(test_config, dataset_class, device=device)
    elif args.task == 'generate':
        input_str = '<|name|> alimentum <|area|> city centre <|familyfriendly|> no <|begoftext|>'
        generate_from_input(input_str, TestConfig(config), dataset_class, device=device)


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    main()
