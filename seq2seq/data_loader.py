from collections import OrderedDict
import os
import pandas as pd
import re
import regex
from torch.utils.data import Dataset


COMMA_PLACEHOLDER = ' __comma__'


class MRToTextDataset(Dataset):
    """Seq-to-seq dataset with flat structured meaning representation (MR) as input and natural text as output."""
    name = 'mr_to_text'
    delimiters = {}

    def __init__(self, tokenizer, partition='train', lowercase=False, convert_slot_names=False, group_by_mr=False,
                 no_target=False, separate_source_and_target=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.partition = partition

        self.convert_to_lowercase = lowercase
        self.convert_slot_names = convert_slot_names
        self.group_by_mr = group_by_mr
        self.no_target = no_target
        self.separate_source_and_target = separate_source_and_target

        self.mrs = []
        self.mrs_dict = []
        self.mrs_as_list = []
        self.mrs_raw = []
        self.utterances = []

        self.load_data()

    def __len__(self):
        return len(self.mrs)

    def __getitem__(self, idx):
        mr = self.mrs[idx]
        utt = self.utterances[idx] if self.utterances else None

        if self.separate_source_and_target:
            source_str = mr
            if self.no_target:
                target_str = ''
            elif utt is not None:
                target_str = utt
            else:
                raise ValueError('Utterances must be present in training and validation data')
        else:
            if self.no_target:
                # If test set, load the MRs only as source
                source_str = mr + self.bos_token
            elif utt is not None:
                # If training/validation set, concatenate the MR and the utterance with a BOS token in between
                source_str = mr + self.bos_token + utt + self.eos_token
            else:
                raise ValueError('Utterances must be present in training and validation data')
            # When MR and utterance are concatenated as source, use the target string as an auxiliary variable
            target_str = mr + self.bos_token

        return source_str, target_str

    def load_data(self):
        # Load the data file
        dataset_path = self.get_data_file_path(self.partition)
        df_data = pd.read_csv(dataset_path, header=0, encoding='utf8')

        self.read_data_from_dataframe(df_data)
        if self.convert_to_lowercase:
            self.lowercase_data()

        # Perform dataset-specific preprocessing of the MRs
        self.mrs_dict, self.mrs_as_list = zip(*[self.preprocess_mr(mr) for mr in self.mrs])
        # self.mrs = [self.convert_mr_dict_to_str(mr_dict) for mr_dict in self.mrs_dict]
        self.mrs = [self.convert_mr_list_to_str(mr_as_list) for mr_as_list in self.mrs_as_list]

        # DEBUG
        # print('>> MRs:\n{}'.format('\n'.join(self.mrs[:50])))
        # if isinstance(self.utterances[0], str):
        #     print('>> Utterances:\n{}'.format('\n'.join(self.utterances[:10])))
        # else:
        #     print('>> Utterances:\n{}'.format('\n'.join(['[' + '; '.join(utt) + ']' for utt in self.utterances[:10]])))

        if self.utterances:
            assert len(self.mrs) == len(self.utterances)

        # DEBUG
        # self.mrs = self.mrs[:10]
        # self.mrs_raw = self.mrs_raw[:10]
        # self.mrs_dict = self.mrs_dict[:10]
        # self.utterances = self.utterances[:10]

    @staticmethod
    def get_data_file_path(partition):
        raise NotImplementedError('method \'get_data_file_path\' must be defined by subclass')

    def read_data_from_dataframe(self, df_data):
        # Extract the column names
        mr_col_name = df_data.columns[0]
        utt_col_name = df_data.columns[1] if df_data.shape[1] > 1 else None

        # Save the MRs and the utterances as lists (repeated MRs are collapsed for test data)
        if self.group_by_mr:
            # If utterances are present in the data
            if df_data.shape[1] > 1:
                # Group by MR, and aggregate utterances into lists
                df_grouped_by_mr = df_data.groupby(mr_col_name, sort=False)[utt_col_name].apply(list).reset_index()
                self.mrs = df_grouped_by_mr[mr_col_name].tolist()
                self.utterances = df_grouped_by_mr[utt_col_name].tolist()

                # mr_list = df_data[mr_col_name].tolist()
                # utt_list = df_data[utt_col_name].tolist()
                # cur_utt_group = []
                #
                # for i in range(len(mr_list)):
                #     if i > 0 and mr_list[i] != mr_list[i - 1]:
                #         self.mrs.append(mr_list[i - 1])
                #         self.utterances.append(cur_utt_group)
                #         cur_utt_group = []
                #     cur_utt_group.append(utt_list[i])
                # self.mrs.append(mr_list[-1])
                # self.utterances.append(cur_utt_group)
            else:
                self.mrs = df_data[mr_col_name].tolist()
        else:
            self.mrs = df_data[mr_col_name].tolist()
            if df_data.shape[1] > 1:
                self.utterances = df_data[utt_col_name].tolist()
            else:
                raise ValueError('Training and validation input data are expected to have two columns')

        # Store original MRs (before any preprocessing)
        self.mrs_raw = self.mrs[:]

    def preprocess_mr(self, mr_str):
        mr_dict = OrderedDict()
        mr_list = []

        mr_str = self.preprocess_da_in_mr(mr_str)

        # Replace commas in values if comma is the slot separator
        if self.delimiters['slot_sep'].strip() == ',' and self.delimiters.get('val_end') is not None:
            mr_str = self.replace_commas_in_slot_values(mr_str, self.delimiters['val_beg'], self.delimiters['val_end'])

        # Extract the sequence of slots and their corresponding values
        for slot_value_pair in mr_str.split(self.delimiters['slot_sep']):
            slot, value = self.parse_slot_and_value(slot_value_pair)
            if self.convert_slot_names:
                slot = self.convert_slot_name(slot)

            # if slot in mr_dict:
            #     slot_new = slot
            #     slot_ctr = 1
            #     while slot_new in mr_dict:
            #         slot_new = slot + str(slot_ctr)
            #         slot_ctr += 1
            #     slot = slot_new

            mr_dict[slot] = value
            mr_list.append((slot, value))

        return mr_dict, mr_list

    @staticmethod
    def convert_mr_dict_to_str(mr_dict):
        return ' '.join(['{0}{1}'.format(slot, ' ' + val if val else '') for slot, val in mr_dict.items()])

    @staticmethod
    def convert_mr_list_to_str(mr_list):
        return ' '.join(['{0}{1}'.format(slot, ' ' + val if val else '') for slot, val in mr_list])

    @classmethod
    def preprocess_da_in_mr(cls, mr):
        """Converts the DA type indication(s) in the MR to the slot-and-value format."""

        # If no DA type indication is expected in the data, return the MR unchanged
        if cls.delimiters.get('da_beg') is None:
            return mr

        mr_new = ''

        if cls.delimiters.get('da_sep') is None:
            # Parse a single DA with its slots and values
            match = regex.match(r'(\S+){0}(.*?){1}$'.format(
                re.escape(cls.delimiters['da_beg']), re.escape(cls.delimiters['da_end'])), mr)
        else:
            # Parse multiple DAs with their respective slots and values
            match = regex.match(r'(\S+){0}(.*?){1}(?:{2}(\S+){0}(.*?){1})*'.format(
                re.escape(cls.delimiters['da_beg']), re.escape(cls.delimiters['da_end']),
                re.escape(cls.delimiters['da_sep'])), mr)

        if not match:
            print(f'Warning: Unexpected format of the following MR:\n{mr}')
            return mr

        for i in range(1, len(match.groups()) + 1, 2):
            if match.group(i) is None:
                break

            for j in range(len(match.captures(i))):
                da_type = match.captures(i)[j]
                slot_value_pairs = match.captures(i + 1)[j]

                if i > 1:
                    mr_new += cls.delimiters['da_sep']

                # Convert the extracted DA type to the slot-value form and prepend it to the DA's slots and values
                mr_new += 'da' + cls.delimiters['val_beg'] + da_type
                if cls.delimiters.get('val_end') is not None:
                    mr_new += cls.delimiters['val_end']
                if len(slot_value_pairs) > 0:
                    mr_new += cls.delimiters['slot_sep'] + slot_value_pairs

        return mr_new

    @classmethod
    def parse_slot_and_value(cls, slot_value_pair_str):
        """Parses out the slot name and the slot value from a string representing this pair."""
        delim_idx = slot_value_pair_str.find(cls.delimiters['val_beg'])
        if delim_idx > -1:
            # Parse the slot
            slot = slot_value_pair_str[:delim_idx].strip()
            # Parse the value
            if cls.delimiters.get('val_end') is not None:
                value = slot_value_pair_str[delim_idx + 1:-1].strip()
            else:
                value = slot_value_pair_str[delim_idx + 1:].strip()
        else:
            # Parse the slot
            if cls.delimiters.get('val_end') is not None:
                slot = slot_value_pair_str[:-1].strip()
            else:
                slot = slot_value_pair_str.strip()
            # Set the value to the empty string
            value = ''

        slot_processed = slot.replace(' ', '').lower()
        value = value.replace(COMMA_PLACEHOLDER, ',')

        return slot_processed, value

    @staticmethod
    def replace_commas_in_slot_values(mr, val_sep, val_sep_end):
        mr_new = ''
        val_beg_cnt = 0
        val_end_cnt = 0

        for c in mr:
            # If comma inside a value, replace the comma with placeholder
            if c == ',' and val_beg_cnt > val_end_cnt:
                mr_new += COMMA_PLACEHOLDER
                continue

            # Keep track of value beginning and end
            if c == val_sep:
                val_beg_cnt += 1
            elif c == val_sep_end:
                val_end_cnt += 1

            mr_new += c

        return mr_new

    @staticmethod
    def put_back_commas_in_mr_values(mrs):
        return [mr.replace(COMMA_PLACEHOLDER, ',') for mr in mrs]

    def lowercase_data(self):
        """Lowercases all MRs and utterances."""
        self.mrs = [mr.lower() for mr in self.mrs]
        if self.utterances:
            if isinstance(self.utterances[0], str):
                self.utterances = [utt.lower() for utt in self.utterances]
            elif isinstance(self.utterances[0], list):
                self.utterances = [[utt.lower() for utt in utt_list] for utt_list in self.utterances]

    def create_reference_file_for_testing(self):
        """Creates a text file with groups of utterances corresponding to one MR separated by an empty line."""
        eval_dir = os.path.join('seq2seq', 'eval')
        out_file = os.path.join(eval_dir, 'test_references_{}.txt'.format(self.name))

        with open(out_file, 'w', encoding='utf8') as f_out:
            if isinstance(self.utterances[0], str):
                for i in range(len(self.mrs_raw)):
                    if i > 0 and self.mrs_raw[i] != self.mrs_raw[i - 1]:
                        f_out.write('\n')
                    f_out.write(self.utterances[i] + '\n')
            elif isinstance(self.utterances[0], list):
                for i in range(len(self.utterances)):
                    f_out.write('\n'.join(self.utterances[i]))
                    f_out.write('\n\n')

    def get_mrs(self, raw=False, lowercased=False):
        mrs = self.mrs_raw if raw else self.mrs
        return [mr.lower() for mr in mrs] if lowercased else mrs[:]

    def get_mrs_as_dicts(self, lowercased=False):
        return [mr_dict.lower() for mr_dict in self.mrs_dict] if lowercased else self.mrs_dict[:]

    def get_utterances(self, lowercased=False):
        if lowercased:
            if self.group_by_mr:
                return [[utt.lower() for utt in utt_list] for utt_list in self.utterances]
            else:
                return [utt.lower() for utt in self.utterances]
        else:
            return self.utterances[:]

    @classmethod
    def get_special_tokens(cls, convert_slot_names=False):
        slot_tokens = set()

        if convert_slot_names:
            train_set_path = cls.get_data_file_path('train')
            df_data = pd.read_csv(train_set_path, header=0, encoding='utf8')

            for mr_str in df_data[df_data.columns[0]]:
                mr_str = cls.preprocess_da_in_mr(mr_str)

                # Replace commas in values if comma is the slot separator
                if cls.delimiters['slot_sep'].strip() == ',' and cls.delimiters.get('val_end') is not None:
                    mr_str = cls.replace_commas_in_slot_values(mr_str, cls.delimiters['val_beg'], cls.delimiters['val_end'])

                for slot_value_pair in mr_str.split(cls.delimiters['slot_sep']):
                    slot, _ = cls.parse_slot_and_value(slot_value_pair)
                    slot = cls.convert_slot_name(slot)
                    slot_tokens.add(slot)

        # DEBUG
        # print('>> slot_tokens:', slot_tokens)

        return sorted(list(slot_tokens))

    @staticmethod
    def convert_slot_name(slot_name):
        """Converts a slot name to a special token."""
        return f'<|{slot_name}|>'


class E2EDataset(MRToTextDataset):
    """An MR-to-text dataset in the restaurant domain (provided as part of the E2E NLG Challenge)."""
    name = 'rest_e2e'
    delimiters = {
        'da_beg': None,
        'da_end': None,
        'da_sep': None,
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devset.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'testset.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'trainset.csv')

        return dataset_path


class E2ECleanedDataset(E2EDataset):
    """A cleaned version of the E2E dataset in the restaurant domain."""
    name = 'rest_e2e_cleaned'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e_cleaned')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devel-fixed.no-ol.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test-fixed.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train-fixed.no-ol.csv')

        return dataset_path


class MultiWOZDataset(MRToTextDataset):
    """A multi-domain dataset of task-oriented dialogues."""
    name = 'multiwoz'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': ', ',
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'multiwoz')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path


class ViggoDataset(MRToTextDataset):
    """An MR-to-text dataset in the video game domain."""
    name = 'video_game'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': None,
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path


class ViggoWithE2EDataset(ViggoDataset):
    """The ViGGO dataset with the training set merged with that of the E2E dataset."""
    name = 'video_game'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_with_e2e.csv')
        else:
            dataset_path = super(ViggoWithE2EDataset, ViggoWithE2EDataset).get_data_file_path(partition)

        return dataset_path


class Viggo20Dataset(ViggoDataset):
    """A 20% sample of the ViGGO dataset."""
    name = 'video_game_20'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.2.csv')
        else:
            dataset_path = super(Viggo20Dataset, Viggo20Dataset).get_data_file_path(partition)

        return dataset_path
