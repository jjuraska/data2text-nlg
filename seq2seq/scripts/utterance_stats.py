import argparse
from collections import Counter, OrderedDict
from itertools import chain
import json
from nltk import word_tokenize
import os
import pandas as pd


def extract_utterances_from_file(input_file_path, lowercase=False):
    if input_file_path.endswith('.csv'):
        df_in = pd.read_csv(input_file_path, header=0, encoding='utf8')
        if 'utt' in df_in.columns:
            utterances = df_in['utt'].tolist()
        elif 'ref' in df_in.columns:
            utterances = df_in['ref'].tolist()
        else:
            raise ValueError(f'Neither column "utt" nor "ref" found in {input_file_path}')
    else:
        with open(input_file_path, 'r', encoding='utf8') as f_in:
            utterances = [line.strip() for line in f_in]

    if lowercase:
        utterances = [utt.lower() for utt in utterances]

    return utterances


def utterance_stats(input_file_path, export_vocab=False, verbose=False):
    utterances = extract_utterances_from_file(input_file_path, lowercase=True)

    vocab_ctr = Counter(chain.from_iterable(map(word_tokenize, utterances)))
    vocab_dict = OrderedDict(vocab_ctr.most_common())

    utt_lengths = [len(utt_tokens) for utt_tokens in map(word_tokenize, utterances)]

    vocab_size = len(vocab_dict)
    avg_utt_len = round(sum(utt_lengths) / len(utt_lengths), 2)

    if verbose:
        print('>> Vocabulary size:', vocab_size)
        print('>> Avg. utt. length:', avg_utt_len)
    else:
        print('{}\t{}'.format(vocab_size, avg_utt_len))

    if export_vocab:
        output_file_path = os.path.splitext(input_file_path)[0] + ' [vocab].json'
        with open(output_file_path, 'w', encoding='utf8') as f_out:
            json.dump(vocab_dict, f_out, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Counts the number of unique words in utterances, and calculates the average number of tokens.')
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='file containing utterances (either a text file, or a CSV file with a column "utt"')
    parser.add_argument('-e', '--export_vocab', action='store_true',
                        help='flag indicating that the vocab should be exported to a JSON file (in the same folder)')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print('Error: invalid file path.')
        print('------')
        print('Usage:')
        print('utterance_stats.py -i <file_path> [-e]')
    else:
        utterance_stats(args.input_file, args.export_vocab)
