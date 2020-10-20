import glob
import os
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from transformers import AutoTokenizer


def train_tokenizer(datasets, pretrained_model_name):
    data_file_paths = []
    data_dir = os.path.join('seq2seq', 'data')

    for dataset_name in datasets:
        dataset_dir = os.path.join(data_dir, dataset_name)
        data_file_paths.extend(glob.glob(os.path.join(dataset_dir, '*.csv')))

    print('>> Files found:')
    print('\n'.join(data_file_paths))
    print()

    # Extract special tokens from the pretrained model's tokenizer
    tokenizer_pretrained = AutoTokenizer.from_pretrained(pretrained_model_name)
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

    print('>> Special tokens:')
    print(special_tokens)
    print()

    # Select the tokenizer type based on the pretrained model
    if any(model_name in pretrained_model_name for model_name in ['t5']):
        tokenizer = SentencePieceBPETokenizer()
    else:
        tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=data_file_paths, vocab_size=3000, show_progress=True, special_tokens=special_tokens)

    # Save tokenizer files to disk
    tokenizer_dir = os.path.join('seq2seq', 'tokenizer')
    if '/' in pretrained_model_name:
        pretrained_model_name = pretrained_model_name.split('/')[-1]

    tokenizer.save_model(tokenizer_dir, '{}-{}'.format(pretrained_model_name, '-'.join(datasets)))


if __name__ == '__main__':
    train_tokenizer(['video_game'], 'gpt2')
