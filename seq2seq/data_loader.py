import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRToTextDataset(Dataset):
    """Seq-to-seq dataset with flat structured meaning representation (MR) as input and natural text as output."""
    name = 'mr_to_text'

    def __init__(self, tokenizer, partition='train', lowercase_data=False, convert_slot_names=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.partition = partition

        self.lowercase_data = lowercase_data
        self.convert_slot_names = convert_slot_names

        self.mrs = []
        self.utterances = []

    def __len__(self):
        return len(self.mrs)

    def __getitem__(self, idx):
        mr = self.mrs[idx]
        utt = self.utterances[idx] if self.utterances else None

        if self.partition == 'test':
            # If test set, load the MRs only as inputs
            input_str = mr + self.bos_token
        elif utt is not None:
            # If training/validation set, concatenate the MR and the utterance with a BOS token in between
            input_str = mr + self.bos_token + utt + self.eos_token
        else:
            raise ValueError('Utterances must be present in training and validation data')
        input_str_wo_utt = mr + self.bos_token

        # return self.encode_data(input_str, labels)
        return input_str, input_str_wo_utt

    def preprocess_mr(self, mr):
        raise NotImplementedError('method \'preprocess_mr\' must be defined by subclass')

    def encode_data(self, input_str, labels):
        inputs_encoded = self.tokenizer(input_str, add_special_tokens=False, padding=False, truncation=True, max_length=512)
        input_ids = inputs_encoded['input_ids']
        attention_mask = inputs_encoded['attention_mask']

        labels_encoded = self.tokenizer(labels, add_special_tokens=False, padding=False, truncation=True, max_length=512)
        label_ids = labels_encoded['input_ids']
        label_attention_mask = labels_encoded['attention_mask']

        # Replace padding tokens in the label list with the value -100 (ignored in loss calculation)
        padding_offset = sum(label_attention_mask)
        label_ids = label_ids[:padding_offset] + [-100] * (len(label_attention_mask) - padding_offset)

        # Prefix the label list with a sequence of the value -100 to bring it to the same length as input_ids
        if self.partition == 'test':
            label_offset = len(input_ids)
        else:
            label_offset = len(input_ids) - len(label_ids)
        label_ids = [-100] * label_offset + label_ids
        label_attention_mask = [0] * label_offset + label_attention_mask
        # label_attention_mask = label_attention_mask[:512]

        if self.partition != 'test':
            assert(len(label_ids) == len(input_ids))

        # Add padding
        # input_ids = (input_ids + [self.tokenizer.pad_token_id] * 512)[:512]
        # attention_mask = (attention_mask + [self.tokenizer.pad_token_id] * 512)[:512]
        # label_ids = (label_ids + [self.tokenizer.pad_token_id] * 512)[:512]
        # label_attention_mask = (label_attention_mask + [self.tokenizer.pad_token_id] * 512)[:512]

        # DEBUG
        # print('>> Lengths:', len(input_ids), len(attention_mask), len(label_ids), len(label_attention_mask))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label_ids), torch.tensor(label_attention_mask)

    def get_mrs(self, lowercased=False):
        return [mr.lower() for mr in self.mrs] if lowercased else self.mrs[:]

    def get_utterances(self, lowercased=False):
        return [utt.lower() for utt in self.utterances] if lowercased else self.utterances[:]

    @classmethod
    def get_special_tokens(cls):
        raise NotImplementedError('method \'get_special_tokens\' must be defined by subclass')


class E2EDataset(MRToTextDataset):
    """The MR-to-text dataset in the restaurant domain (provided as part of the E2E NLG Challenge)."""
    name = 'rest_e2e'
    delimiters = {
        'da_beg': None,
        'da_end': None,
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e')
        if self.partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devset.csv')
        elif self.partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'testset.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'trainset.csv')

        df_data = pd.read_csv(dataset_path, header=0, encoding='utf8')

        # Extract the column names
        mr_col_name = df_data.columns[0]
        utt_col_name = df_data.columns[1] if df_data.shape[1] > 1 else None

        # Save the MRs and the utterances as lists (repeated MRs are collapsed for test data)
        if self.partition == 'test':
            if df_data.shape[1] > 1:
                # Group by MR, and aggregate utterances into lists
                df_grouped_by_mr = df_data.groupby(mr_col_name)[utt_col_name].apply(list).reset_index()
                self.mrs = df_grouped_by_mr[mr_col_name].tolist()
                self.utterances = df_grouped_by_mr[utt_col_name].tolist()
            else:
                self.mrs = df_data[mr_col_name].tolist()
        else:
            self.mrs = df_data[mr_col_name].tolist()
            if df_data.shape[1] > 1:
                self.utterances = df_data[utt_col_name].tolist()
            else:
                raise ValueError('Training and validation input data are expected to have two columns')

        # Lowercase all MRs and utterances
        if self.lowercase_data:
            self.mrs = [mr.lower() for mr in self.mrs]
            if self.utterances:
                if isinstance(self.utterances[0], str):
                    self.utterances = [utt.lower() for utt in self.utterances]
                elif isinstance(self.utterances[0], list):
                    self.utterances = [[utt.lower() for utt in utt_list] for utt_list in self.utterances]

        # Perform dataset-specific preprocessing of the MRs
        self.mrs = [self.preprocess_mr(mr) for mr in self.mrs]

        # DEBUG
        # print('>> MRs:\n{}'.format('\n'.join(self.mrs[:10])))
        # print('>> Utterances:\n{}'.format('\n'.join(self.utterances[:10])))

        if self.utterances:
            assert len(self.mrs) == len(self.utterances)

    def preprocess_mr(self, mr_str):
        mr_seq = []

        # Extract the sequence of slots and their corresponding values
        for slot_value_pair in mr_str.split(self.delimiters['slot_sep']):
            slot, value = self.parse_slot_and_value(slot_value_pair)
            if self.convert_slot_names:
                slot = self.convert_slot_name(slot)

            mr_seq.append(slot)
            if value:
                mr_seq.append(value)

        return ' '.join(mr_seq)

    @classmethod
    def parse_slot_and_value(cls, slot_value_pair_str):
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

        return slot_processed, value

    @classmethod
    def get_special_tokens(cls, convert_slot_names=False):
        slot_tokens = set()

        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e')
        train_set_path = os.path.join(dataset_dir, 'trainset.csv')
        df_data = pd.read_csv(train_set_path, header=0, encoding='utf8')

        for mr_str in df_data[df_data.columns[0]]:
            for slot_value_pair in mr_str.split(cls.delimiters['slot_sep']):
                slot, _ = cls.parse_slot_and_value(slot_value_pair)
                slot = cls.convert_slot_name(slot)
                slot_tokens.add(slot)

        return list(slot_tokens)

    @staticmethod
    def convert_slot_name(slot_name):
        """Converts a slot name to a special token."""
        return f'<|{slot_name}|>'
