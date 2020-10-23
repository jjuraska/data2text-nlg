import os
import sys
# from tokenizers.implementations import ByteLevelBPETokenizer, SentencePieceBPETokenizer
import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_bart import shift_tokens_right
import yaml


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


def load_model_and_tokenizer(config, special_tokens=None):
    if 'bart' in config.model_name:
        loading_function = load_bart_model_and_tokenizer
    elif 'gpt2' in config.model_name:
        loading_function = load_gpt2_model_and_tokenizer
    elif 't5' in config.model_name:
        loading_function = load_t5_model_and_tokenizer
    else:
        print('Error: model "{}" not supported'.format(config.model_name))
        sys.exit()

    model, tokenizer = loading_function(config, special_tokens=special_tokens)

    if config.checkpoint_epoch is not None and config.checkpoint_step is not None:
        load_model_checkpoint(model, config.model_name, config.checkpoint_epoch, config.checkpoint_step)

    assert model.config.vocab_size == len(tokenizer), \
        'Model\'s vocab size ({}) does not match tokenizer\'s vocab size ({})'.format(
               model.config.vocab_size, len(tokenizer))

    # Special tokens summary
    print('>> Tokenizer\'s special tokens:')
    print('bos_token: {} (ID: {})'.format(tokenizer.bos_token, tokenizer.bos_token_id))
    print('eos_token: {} (ID: {})'.format(tokenizer.eos_token, tokenizer.eos_token_id))
    print('pad_token: {} (ID: {})'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    print('unk_token: {} (ID: {})'.format(tokenizer.unk_token, tokenizer.unk_token_id))
    print()

    print('>> Model\'s special tokens:')
    print('bos_token ID: {}'.format(model.config.bos_token_id))
    print('eos_token ID: {}'.format(model.config.eos_token_id))
    print('pad_token ID: {}'.format(model.config.pad_token_id))
    if model.config.is_encoder_decoder:
        print('decoder_start_token ID: {}'.format(model.config.decoder_start_token_id))
    print()

    print('>> Tokenizer\'s additional special tokens:')
    print('\n'.join(f'{tok}: {tok_id}' for tok, tok_id in tokenizer.get_added_vocab().items()))
    print()

    return model, tokenizer


def load_bart_model_and_tokenizer(config, special_tokens=None):
    # Specify which additional special tokens should never be split by the tokenizer
    special_tokens_dict = {
        'additional_special_tokens': special_tokens
    }

    if config.pretrained or config.tokenizer_name is None:
        tokenizer = BartTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = load_custom_tokenizer(BartTokenizer, config.tokenizer_name, config.max_seq_length)
        special_tokens_dict['bos_token'] = tokenizer.bos_token
        special_tokens_dict['eos_token'] = tokenizer.eos_token
        special_tokens_dict['pad_token'] = tokenizer.pad_token
        special_tokens_dict['unk_token'] = tokenizer.unk_token

    tokenizer.add_special_tokens(special_tokens_dict)

    if config.pretrained:
        # Load model with pretrained weights
        model = BartForConditionalGeneration.from_pretrained(config.model_name)
    else:
        # Load model without pretrained weights
        config = BartConfig.from_pretrained(config.model_name, vocab_size=len(tokenizer))
        model = BartForConditionalGeneration(config)

        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.unk_token_id = tokenizer.unk_token_id

    # Resize the model's embedding matrix to accommodate the added special tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_gpt2_model_and_tokenizer(config, special_tokens=None):
    # Specify which additional special tokens should never be split by the tokenizer
    special_tokens_dict = {
        'bos_token': '<|begoftext|>',
        'pad_token': '<pad>',
        'additional_special_tokens': special_tokens
    }

    if config.pretrained or config.tokenizer_name is None:
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = load_custom_tokenizer(GPT2Tokenizer, config.tokenizer_name, config.max_seq_length)
        special_tokens_dict['eos_token'] = tokenizer.eos_token

    tokenizer.add_special_tokens(special_tokens_dict)

    if config.pretrained:
        # Load model with pretrained weights
        model = GPT2LMHeadModel.from_pretrained(config.model_name)
    else:
        # Load model without pretrained weights
        config = GPT2Config.from_pretrained(config.model_name)
        model = GPT2LMHeadModel(config)

        model.config.eos_token_id = tokenizer.eos_token_id

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Resize the model's embedding matrix to accommodate the added special tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_t5_model_and_tokenizer(config, special_tokens=None):
    # Specify which additional special tokens should never be split by the tokenizer
    special_tokens_dict = {
        'additional_special_tokens': special_tokens
    }

    # if config.pretrained or config.tokenizer_name is None:
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    # else:
    #     tokenizer = load_custom_tokenizer(SentencePieceBPETokenizer, config.tokenizer_name, config.max_seq_length)
    #     special_tokens_dict['eos_token'] = tokenizer.eos_token
    #     special_tokens_dict['pad_token'] = tokenizer.pad_token
    #     special_tokens_dict['unk_token'] = tokenizer.unk_token

    tokenizer.add_special_tokens(special_tokens_dict)

    if config.pretrained:
        # Load model with pretrained weights
        model = T5ForConditionalGeneration.from_pretrained(config.model_name)

        # Resize the model's embedding matrix to accommodate the added special tokens
        model.resize_token_embeddings(len(tokenizer))
    else:
        # Load model without pretrained weights
        config = T5Config.from_pretrained(config.model_name, vocab_size=len(tokenizer))
        model = T5ForConditionalGeneration(config)

        # Resize the model's embedding matrix to accommodate the added special tokens
        # model.resize_token_embeddings(tokenizer.get_vocab_size())
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_custom_tokenizer(tokenizer_class, tokenizer_name, max_length):
    """Loads a tokenizer trained on a custom dataset(s)."""

    tokenizer_dir = os.path.join('seq2seq', 'tokenizer')
    vocab_file = os.path.join(tokenizer_dir, f'{tokenizer_name}-vocab.json')
    merges_file = os.path.join(tokenizer_dir, f'{tokenizer_name}-merges.txt')

    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise FileNotFoundError(f'Tokenizer files not found in the "{tokenizer_dir}" directory')

    return tokenizer_class(vocab_file, merges_file, model_max_length=max_length)


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


def prepare_batch(config, batch, tokenizer, is_enc_dec):
    batch_dict = {}

    # DEBUG
    # print()
    # print('>> SOURCES:\n', '\n'.join(batch[0]))
    # print()
    # print('>> TARGETS:\n', '\n'.join(batch[1]))
    # print()

    # TODO: Incorporate into the data loader?
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

        if 'bart' in config.model_name:
            """Prepare decoder inputs manually because BART gets confused by the -100 mask values during
            automatic generation of decoder inputs from labels, expecting the padding token IDs instead."""
            decoder_input_ids = shift_tokens_right(targets['input_ids'], tokenizer.pad_token_id)
            # decoder_mask = targets['attention_mask']
        else:
            # Decoder input IDs and mask are inferred automatically from labels
            decoder_input_ids = None

        batch_dict['decoder_input_ids'] = decoder_input_ids
    else:
        # if 'gpt2' in config.model_name:
        #     tokenizer.padding_side = 'left'

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

    batch_dict['input_ids'] = input_ids
    batch_dict['attention_mask'] = input_mask
    batch_dict['labels'] = label_ids

    return batch_dict


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
