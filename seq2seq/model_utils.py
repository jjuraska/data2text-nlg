import os
import sys
# from tokenizers.implementations import ByteLevelBPETokenizer, SentencePieceBPETokenizer
import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
import yaml


class GPT2LMHeadModelWithTokenTypes(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, expand_token_type_size=1, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if token_type_ids is not None and expand_token_type_size > 1:
            expanded_return_idx = (
                torch.arange(token_type_ids.shape[0]).view(-1, 1).repeat(1, expand_token_type_size).view(-1).to(token_type_ids.device)
            )
            token_type_ids = token_type_ids.index_select(0, expanded_return_idx)

        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


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

    if config.model_name.count('/') < 2 and config.checkpoint_epoch is not None and config.checkpoint_step is not None:
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
        model = GPT2LMHeadModelWithTokenTypes.from_pretrained(config.model_name)
    else:
        # Load model without pretrained weights
        config = GPT2Config.from_pretrained(config.model_name)
        model = GPT2LMHeadModelWithTokenTypes(config)

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

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.unk_token_id = tokenizer.unk_token_id

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


def save_model(model, tokenizer, epoch, step):
    model_dir = os.path.join('seq2seq', 'model', f'epoch_{epoch}_step_{step}')
    os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def save_model_checkpoint(model, model_name, epoch, step):
    model_dir = os.path.join('seq2seq', 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    file_name = '{}_epoch_{}_step_{}.pt'.format(model_name, epoch, step)
    torch.save(model.state_dict(), os.path.join(model_dir, file_name))


def prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=False, bool_slots=None):
    batch_dict = {}

    # DEBUG
    # print()
    # print('>> SOURCES:\n', '\n'.join(batch[0]))
    # print()
    # print('>> TARGETS:\n', '\n'.join(batch[1]))
    # print()

    # TODO: Incorporate into the data loader?
    if is_enc_dec:
        inputs = tokenizer(batch[0], add_special_tokens=True, max_length=config.max_seq_length,
                           padding=True, truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']

        if include_labels:
            labels = tokenizer(batch[1], add_special_tokens=True, max_length=config.max_seq_length,
                               padding=True, truncation=True, return_tensors='pt')

            label_ids = labels['input_ids'].clone()
            label_ids[label_ids == tokenizer.pad_token_id] = -100

            if 'bart' in config.model_name:
                """Prepare decoder inputs manually because BART gets confused by the -100 mask values during
                automatic generation of decoder inputs from labels, expecting the padding token IDs instead. 
                T5 infers decoder input IDs and mask automatically from labels."""
                batch_dict['decoder_input_ids'] = shift_tokens_right(labels['input_ids'], tokenizer.pad_token_id)
                # batch_dict['decoder_mask'] = labels['attention_mask']

        if bool_slots:
            batch_dict['slot_spans'] = get_slot_spans(input_ids, bool_slots, tokenizer)
    else:
        inputs = tokenizer(batch[0], add_special_tokens=False, max_length=config.max_seq_length,
                           padding=True, truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']

        if 'gpt2' in config.model_name and config.use_token_type_ids:
            batch_dict['token_type_ids'] = create_token_type_ids(input_ids, batch[2], tokenizer)

        if include_labels:
            mrs_only = tokenizer(batch[1], add_special_tokens=False, max_length=config.max_seq_length,
                                 padding=True, truncation=True, return_tensors='pt')

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
    # print('>> Token type IDs:\n', batch_dict['token_type_ids'])
    # print()
    # print('>> ENCODED labels:\n', label_ids)
    # print('>> DECODED labels:\n', tokenizer.decode(label_ids[0][label_ids[0] >= 0]))
    # print()
    # print('>> ENCODED decoder masks:\n', decoder_mask)
    # print()

    batch_dict['input_ids'] = input_ids
    batch_dict['attention_mask'] = input_mask
    if include_labels:
        batch_dict['labels'] = label_ids

    return batch_dict


def create_token_type_ids(input_id_batch, token_type_seq_batch, tokenizer):
    """Creates a token type ID tensor for the given batch of input IDs. (Only works with the GPT-2 model.)

    If slot names are converted to special tokens, the token types are calculated by carrying over the ID of the most
    recent special token in the input sequence (e.g., a slot name or the BOS token). If slot names are not converted to
    special tokens (and can thus comprise multiple tokens), token type sequences pre-calculated using dataset-specific
    mapping of slot names to salient tokens are used instead.
    """
    token_type_id_batch = input_id_batch.clone()

    if not token_type_seq_batch[0]:
        # Prepare the special token ID set containing tokenizer's special tokens, plus slot names (as special tokens)
        special_token_id_set = {val for key, val in tokenizer.get_added_vocab().items()}
        special_token_id_set.update([tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id])

        # DEBUG
        # print('>> Special token ID set:')
        # print(special_token_id_set)

        for i in range(token_type_id_batch.shape[0]):
            prev_special_token = token_type_id_batch[i][0]
            for j in range(1, token_type_id_batch.shape[1]):
                if token_type_id_batch[i][j].item() in special_token_id_set or j == 0:
                    # Set the current token as the most recent special token
                    prev_special_token = token_type_id_batch[i][j].item()
                else:
                    # Carry over the most recent special token
                    token_type_id_batch[i][j] = prev_special_token
    else:
        # Get the ID of the slot separator token used in the input sequences
        slot_sep_id = tokenizer(' |')['input_ids'][0]

        for i in range(token_type_id_batch.shape[0]):
            cur_token_type_idx = 0
            token_types = tokenizer(token_type_seq_batch[i], add_special_tokens=False)['input_ids']

            assert len(token_types) == len(token_type_seq_batch[i].split())

            # Skip padding tokens (if input IDs padded from the left)
            first_non_pad_idx = 0
            while token_type_id_batch[i][first_non_pad_idx].item() == tokenizer.pad_token_id:
                first_non_pad_idx += 1

            for j in range(first_non_pad_idx, token_type_id_batch.shape[1]):
                if token_type_id_batch[i][j].item() == tokenizer.eos_token_id:
                    break
                elif token_type_id_batch[i][j].item() in [slot_sep_id, tokenizer.bos_token_id]:
                    # Set the token type to the next salient token in the pre-calculated sequence
                    cur_token_type_idx += 1

                # Carry over the token type until the next slot separator is encountered in the input sequence
                token_type_id_batch[i][j] = token_types[cur_token_type_idx]

    return token_type_id_batch


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


def get_slot_spans(input_id_batch, bool_slots, tokenizer):
    slot_span_batch = []

    for i, input_ids in enumerate(input_id_batch):
        input_tokens = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in input_ids]

        slot_spans = []
        cur_name_beg = 0
        cur_value_beg = 0
        cur_slot = {}

        for tok_pos, tok in enumerate(input_tokens):
            # TODO: skip BOS token
            tok_stripped = tok.strip()
            if tok_stripped == '=':
                cur_slot['name_span'] = (cur_name_beg, tok_pos - 1)
                cur_slot['value_span'] = []
                cur_value_beg = tok_pos + 1
            # TODO: only do this for list-slots to avoid false positives (e.g., in address slots)
            elif tok_stripped == ',':
                cur_slot['value_span'].append((cur_value_beg, tok_pos - 1))
                cur_value_beg = tok_pos + 1
            elif tok_stripped in ['|', tokenizer.eos_token]:
                if cur_value_beg > cur_name_beg:
                    cur_slot['value_span'].append((cur_value_beg, tok_pos - 1))
                else:
                    cur_slot['name_span'] = (cur_name_beg, tok_pos - 1)

                # Ignore non-content slots, and mark Boolean slots
                slot_name = tokenizer.decode(input_ids[cur_slot['name_span'][0]:cur_slot['name_span'][1] + 1]).strip()
                if slot_name not in {'intent', 'topic'}:
                    cur_slot['name'] = slot_name
                    cur_slot['is_boolean'] = slot_name in bool_slots
                    num_value_elements = max(1, len(cur_slot.get('value_span', [])))
                    cur_slot['mentioned'] = [False] * num_value_elements
                    cur_slot['confidence'] = [False] * num_value_elements
                    slot_spans.append(cur_slot)

                cur_name_beg = tok_pos + 1
                cur_slot = {}

        slot_span_batch.append(slot_spans)

        # DEBUG
        # print('>> input tokens:', input_tokens)
        # print('>> slot spans:', slot_spans)
        # print()

    return slot_span_batch
