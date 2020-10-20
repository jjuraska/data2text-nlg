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
    if 'gpt2' in config.model_name:
        loading_function = load_pretrained_gpt2_model_and_tokenizer
        is_enc_dec = False
    elif 'bart' in config.model_name:
        loading_function = load_pretrained_bart_model_and_tokenizer
        is_enc_dec = True
    elif 't5' in config.model_name:
        loading_function = load_pretrained_t5_model_and_tokenizer
        is_enc_dec = True
    else:
        print('Error: model "{}" not supported'.format(config.model_name))
        sys.exit()

    model, tokenizer = loading_function(config.model_name, pretrained=config.pretrained, special_tokens=special_tokens)

    if config.checkpoint_epoch is not None and config.checkpoint_step is not None:
        load_model_checkpoint(model, config.model_name, config.checkpoint_epoch, config.checkpoint_step)

    return model, tokenizer, is_enc_dec


def load_pretrained_bart_model_and_tokenizer(model_name, pretrained=False, special_tokens=None):
    if pretrained:
        # Load pretrained tokenizer
        tokenizer = BartTokenizer.from_pretrained(model_name)
    else:
        # Load tokenizer trained on custom dataset(s)
        tokenizer_dir = os.path.join('seq2seq', 'tokenizer')
        tokenizer = BartTokenizer(os.path.join(tokenizer_dir, 'bart-base-video_game-vocab.json'),
                                  os.path.join(tokenizer_dir, 'bart-base-video_game-merges.txt'),
                                  model_max_length=64)

    special_tokens = {
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    if pretrained:
        # Load model with pretrained weights
        model = BartForConditionalGeneration.from_pretrained(model_name)
    else:
        # Load model without pretrained weights
        config = BartConfig.from_pretrained(model_name, vocab_size=tokenizer.vocab_size)
        model = BartForConditionalGeneration(config)
        print('>> config.vocab_size:', config.vocab_size)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_pretrained_gpt2_model_and_tokenizer(model_name, pretrained=False, special_tokens=None):
    if pretrained:
        # Load pretrained tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        # Load tokenizer trained on custom dataset(s)
        tokenizer_dir = os.path.join('seq2seq', 'tokenizer')
        tokenizer = GPT2Tokenizer(os.path.join(tokenizer_dir, 'gpt2-video_game-vocab.json'),
                                  os.path.join(tokenizer_dir, 'gpt2-video_game-merges.txt'),
                                  model_max_length=64)

    special_tokens = {
        'bos_token': '<|begoftext|>',
        'pad_token': '<pad>',
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    if pretrained:
        # Load model with pretrained weights
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        # Load model without pretrained weights
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_pretrained_t5_model_and_tokenizer(model_name, pretrained=False, special_tokens=None):
    # if pretrained:
    # Load pretrained tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    special_tokens = {
        'additional_special_tokens': special_tokens
    }
    # else:
    #     tokenizer_dir = os.path.join('seq2seq', 'tokenizer')
    #     tokenizer = SentencePieceBPETokenizer.from_file(os.path.join(tokenizer_dir, 't5-video_game-vocab.json'),
    #                                                     os.path.join(tokenizer_dir, 't5-video_game-merges.txt'))

    tokenizer.add_special_tokens(special_tokens)

    if pretrained:
        # Load model with pretrained weights
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        # Load model without pretrained weights
        config = T5Config.from_pretrained(model_name, vocab_size=tokenizer.vocab_size)
        model = T5ForConditionalGeneration(config)
        # model.resize_token_embeddings(tokenizer.get_vocab_size())
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


def prepare_batch(config, batch, tokenizer, is_enc_dec, device='cpu'):
    batch_dict = {}

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
        # tokenizer.padding_side = 'left'

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

    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': input_mask,
        'labels': label_ids
    }

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
