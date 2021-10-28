import os
import sys
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from transformers import AutoConfig, AutoTokenizer

from data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset


def train_tokenizer(datasets, pretrained_model_name, vocab_size=1000, lowercase=False, convert_slot_names=False):
    train_data = []
    data_file_paths = []

    # Create the tokenizer directory if it does not exist
    tokenizer_data_dir = os.path.join('tokenizer', 'data_files')
    if not os.path.exists(tokenizer_data_dir):
        os.makedirs(tokenizer_data_dir)

    # Load the pretrained model's configuration and tokenizer
    print('Loading pretrained model\'s tokenizer...', end='')
    sys.stdout.flush()
    config_pretrained = AutoConfig.from_pretrained(pretrained_model_name)
    tokenizer_pretrained = AutoTokenizer.from_pretrained(pretrained_model_name)
    print(' Done')

    # Extract special tokens from the pretrained model's tokenizer
    special_tokens = [
        (tokenizer_pretrained.pad_token, tokenizer_pretrained.pad_token_id),
        (tokenizer_pretrained.bos_token, tokenizer_pretrained.bos_token_id),
        (tokenizer_pretrained.eos_token, tokenizer_pretrained.eos_token_id),
        (tokenizer_pretrained.unk_token, tokenizer_pretrained.unk_token_id),
        (tokenizer_pretrained.sep_token, tokenizer_pretrained.sep_token_id),
        (tokenizer_pretrained.cls_token, tokenizer_pretrained.cls_token_id),
        # (tokenizer_pretrained.mask_token, tokenizer_pretrained.mask_token_id),
    ]

    # Sort special tokens by their ID in the pretrained model's tokenizer
    special_tokens = [token for token in set(special_tokens) if token[1] is not None]
    special_tokens = [token[0] for token in sorted(special_tokens, key=lambda x: x[1])]

    # Get the corresponding dataset class
    datasets.sort()
    for dataset in datasets:
        if dataset == 'rest_e2e':
            dataset_class = E2EDataset
        elif dataset == 'rest_e2e_cleaned':
            dataset_class = E2ECleanedDataset
        elif dataset == 'multiwoz':
            dataset_class = MultiWOZDataset
        elif dataset == 'video_game':
            dataset_class = ViggoDataset
        else:
            print('Error: dataset "{}" not recognized'.format(dataset))
            sys.exit()

        train_set = dataset_class(tokenizer_pretrained, partition='train', lowercase=lowercase,
                                  convert_slot_names=convert_slot_names,
                                  separate_source_and_target=config_pretrained.is_encoder_decoder)

        mrs = [train_set.convert_mr_from_list_to_str(mr, add_separators=(not convert_slot_names))
               for mr in train_set.get_mrs(lowercase=lowercase, convert_slot_names=convert_slot_names)]
        train_data.extend(mrs)
        train_data.extend(train_set.get_utterances(lowercase=lowercase))

        special_tokens.extend(train_set.get_special_tokens(convert_slot_names=convert_slot_names))

        # Write the processed data to a simple text file
        data_file_path = os.path.join(tokenizer_data_dir, dataset + '.txt')
        with open(data_file_path, 'w') as f_data:
            f_data.write('\n'.join(train_data))

        data_file_paths.append(data_file_path)

    print('>> Special tokens:')
    print(special_tokens)
    print()

    # Determine the tokenizer type based on the pretrained model
    if any(model_name in pretrained_model_name for model_name in ['t5']):
        tokenizer = SentencePieceBPETokenizer()
    else:
        tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=data_file_paths, vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens)

    # Save tokenizer files to disk
    if '/' in pretrained_model_name:
        pretrained_model_name = pretrained_model_name.split('/')[-1]
    file_name = '{}-{}'.format(pretrained_model_name, '-'.join(datasets))
    if lowercase:
        file_name += '-lowercase'
    tokenizer.save_model(tokenizer_dir, file_name)


if __name__ == '__main__':
    train_tokenizer(['multiwoz'], 'facebook/bart-base', vocab_size=10000, lowercase=False, convert_slot_names=False)
