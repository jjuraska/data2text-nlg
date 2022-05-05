import argparse
from collections import Counter, OrderedDict
from itertools import chain
import json
from nltk import word_tokenize
from nltk.util import bigrams
import os
import pandas as pd

from data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset


def extract_utterances_from_file(input_file_path, lowercase=False):
    if input_file_path.endswith('.csv'):
        df_in = pd.read_csv(input_file_path, header=0, encoding='utf8')
        if 'utt' in df_in.columns:
            utterances = df_in['utt'].fillna('').tolist()
        elif 'ref' in df_in.columns:
            utterances = df_in['ref'].fillna('').tolist()
        else:
            raise ValueError(f'Neither column "utt" nor "ref" found in {input_file_path}')
    else:
        with open(input_file_path, 'r', encoding='utf8') as f_in:
            utterances = [line.strip() for line in f_in]

    if lowercase:
        utterances = [utt.lower() if isinstance(utt, str) else '' for utt in utterances]

    return utterances


def extract_mrs_from_file(input_file_path):
    if input_file_path.endswith('.csv'):
        df_in = pd.read_csv(input_file_path, header=0, encoding='utf8')
        if 'mr' in df_in.columns:
            mrs = df_in['mr'].tolist()
        elif 'input' in df_in.columns:
            mrs = df_in['input'].tolist()
        else:
            raise ValueError(f'Neither column "mr" nor "input" found in {input_file_path}')
    else:
        with open(input_file_path, 'r', encoding='utf8') as f_in:
            mrs = [line.strip() for line in f_in]

    return mrs


def count_unique_elements_in_counter(vocab_freq_ctr):
    vocab_freq_list = list(vocab_freq_ctr.items())
    assert vocab_freq_ctr and vocab_freq_list[0][1] >= vocab_freq_list[-1][1],\
        'Vocab dictionary must be sorted in a descending order of frequency.'

    num_unique_elements = 0

    for w, cnt in vocab_freq_list[::-1]:
        if cnt == 0:
            continue
        if cnt > 1:
            break
        num_unique_elements += 1

    return num_unique_elements


def calculate_utterance_stats(utterances):
    utterances_tok = list(map(word_tokenize, utterances))

    # Calculate unigram statistics
    vocab_ctr = Counter(chain.from_iterable(utterances_tok))
    vocab_dict = OrderedDict(vocab_ctr.most_common())
    vocab_size = len(vocab_dict)
    num_unique_words = count_unique_elements_in_counter(vocab_dict)

    # Calculate bigram statistics
    bigram_ctr = Counter(chain.from_iterable(map(bigrams, utterances_tok)))
    bigram_dict = OrderedDict(bigram_ctr.most_common())
    bigram_vocab_size = len(bigram_dict)
    num_unique_bigrams = count_unique_elements_in_counter(bigram_dict)
    # Convert keys from tuples to strings for export
    bigram_dict = OrderedDict([(f'{key_tuple[0]} {key_tuple[1]}', value)
                               for key_tuple, value in bigram_dict.items()])

    # Calculate the average utterance length
    utt_lengths = [len(utt_tokens) for utt_tokens in utterances_tok]
    avg_utt_len = round(sum(utt_lengths) / len(utt_lengths), 2)

    # Calculate the proportion of unique utterances
    num_unique_utterances = len(set(utterances))
    unique_utterance_proportion = round(num_unique_utterances / len(utterances), 4)

    stats = OrderedDict([
        ('Vocab', vocab_size),
        ('Bigram vocab', bigram_vocab_size),
        ('Distinct-1', round(vocab_size / sum(vocab_ctr.values()), 4)),
        ('Distinct-2', round(bigram_vocab_size / sum(bigram_ctr.values()), 4)),
        ('Unique-1', num_unique_words),
        ('Unique-2', num_unique_bigrams),
        ('Avg. utt. length', avg_utt_len),
        ('Unique utt. proportion', f'{round(unique_utterance_proportion * 100, 2)}%'),
    ])
    vocabs = OrderedDict([
        ('unigram', vocab_dict),
        ('bigram', bigram_dict),
    ])

    return stats, vocabs


def get_utterance_stats(utt_file_path, mr_file_path=None, dataset_class=None, export_delex=False, export_vocab=False,
                        verbose=False):
    utterances = extract_utterances_from_file(utt_file_path, lowercase=True)
    should_delexicalize = mr_file_path is not None and dataset_class is not None

    if should_delexicalize:
        # Extract MRs and delexicalize the utterances
        mrs_raw = extract_mrs_from_file(mr_file_path)
        utterances = dataset_class.delexicalize_utterances(mrs_raw, utterances, lowercase=True)

        if export_delex:
            df_out = pd.DataFrame({
                'mr': mrs_raw,
                'utt': utterances
            })
            out_file_path = os.path.splitext(utt_file_path)[0] + ' [delex].csv'
            df_out.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    stats, vocabs = calculate_utterance_stats(utterances)

    # Print calculated utterance stats
    if verbose:
        for metric, value in stats.items():
            print(f'>> {metric}: {value}')
    else:
        print('\t'.join([str(value) for value in stats.values()]))

    # Export vocabularies
    if export_vocab:
        delex_suffix = '-delex' if should_delexicalize else ''
        for ngram, vocab_dict in vocabs.items():
            out_file_path = os.path.splitext(utt_file_path)[0] + f' [vocab-{ngram}{delex_suffix}].json'
            with open(out_file_path, 'w', encoding='utf8') as f_out:
                json.dump(vocab_dict, f_out, indent=4, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Counts the number of unique words in utterances, and calculates the average number of tokens.')
    parser.add_argument('-u', '--utterance_file', type=str, required=True,
                        help='File containing utterances (either a text file, or a CSV file with a column "utt" or "ref"')
    parser.add_argument('-m', '--mr_file', type=str,
                        help='File containing input MRs (either a text file, or a CSV file with a column "mr" or "input"')
    parser.add_argument('-d', '--dataset', choices=['rest_e2e', 'rest_e2e_cleaned', 'multiwoz', 'video_game'],
                        help='Dataset name (for utterance delexicalization, mandatory if --mr_file provided)')
    parser.add_argument('-e', '--export_delex', action='store_true',
                        help='Flag indicating that the delexicalized utterances should be exported to a CSV file (in the same folder)')
    parser.add_argument('-v', '--export_vocab', action='store_true',
                        help='Flag indicating that the vocab should be exported to a JSON file (in the same folder)')
    args = parser.parse_args()

    usage_msg = ('------'
                 'Usage:'
                 'utterance_stats.py -u <file_path> [-m <file_path> -d <dataset_name>] [-e] [-v]')

    if not os.path.isfile(args.utterance_file) or (args.mr_file and not os.path.isfile(args.mr_file)):
        print('Error: Invalid file path.')
        print(usage_msg)
    elif args.mr_file and not args.dataset:
        print('Error: Dataset must be specified when providing MRs for delexicalization.')
        print(usage_msg)
    else:
        get_utterance_stats(args.utterance_file, export_vocab=args.export_vocab)

        if args.mr_file:
            if args.dataset == 'rest_e2e':
                dataset_class = E2EDataset
            elif args.dataset == 'rest_e2e_cleaned':
                dataset_class = E2ECleanedDataset
            elif args.dataset == 'multiwoz':
                dataset_class = MultiWOZDataset
            elif args.dataset == 'video_game':
                dataset_class = ViggoDataset
            else:
                raise ValueError(f'Dataset "{args.dataset}" not recognized')

            get_utterance_stats(
                args.utterance_file, mr_file_path=args.mr_file, dataset_class=dataset_class,
                export_delex=args.export_delex, export_vocab=args.export_vocab)


if __name__ == '__main__':
    main()
