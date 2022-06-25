from collections import Counter, defaultdict
import os
import pandas as pd
import random
import re
import regex
from torch.utils.data import Dataset

from constants import SlotNameConversionMode
from ontology import DatasetOntologyBuilder


COMMA_PLACEHOLDER = ' __comma__'


class MRToTextDataset(Dataset):
    """Seq-to-seq dataset with flat structured meaning representation (MR) as input and natural text as output."""
    name = 'mr_to_text'
    delimiters = {}

    def __init__(self, tokenizer, input_str=None, partition='train', lowercase=False, slot_name_conversion=None,
                 group_by_mr=False, no_target=False, separate_source_and_target=False, sort_by_length=False,
                 prepare_token_types=False, num_slot_permutations=0):
        super().__init__()

        # Tokenizer's special tokens
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token

        self.partition = partition

        # Data preprocessing parameters
        self.convert_to_lowercase = lowercase
        self.slot_name_conversion = slot_name_conversion
        self.group_by_mr = group_by_mr
        self.no_target = no_target
        self.separate_source_and_target = separate_source_and_target
        self.sort_by_length = sort_by_length
        self.prepare_token_types = prepare_token_types
        self.num_slot_permutations = num_slot_permutations

        self.mrs_raw = []
        self.mrs_raw_as_lists = []
        self.mrs_as_lists = []
        self.mrs = []
        self.utterances = []

        self.load_data(input_str=input_str)

        ontology_builder = DatasetOntologyBuilder(self, load_from_file=True, preprocess_slot_names=True)
        self.bool_slots = self.identify_boolean_slots(ontology_builder.ontology)

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

        if self.prepare_token_types:
            # The following is required when using slot name verbalization along with token type IDs in the GPT-2 model
            token_type_seq = self.get_token_type_sequence(idx)
            return source_str, target_str, token_type_seq
        else:
            return source_str, target_str

    def load_data(self, input_str=None):
        if input_str:
            self.mrs_raw = [input_str]
            self.utterances = []
        else:
            # Load the data file
            dataset_path = self.get_data_file_path(self.partition)
            df_data = pd.read_csv(dataset_path, header=0, encoding='utf8')

            # Read the MRs and utterances from the file
            self.mrs_raw, self.utterances = self.read_data_from_dataframe(df_data, group_by_mr=self.group_by_mr)

        # Sort data before performing any preprocessing
        if self.sort_by_length:
            self.mrs_raw, self.utterances = self.sort_data(self.mrs_raw, self.utterances, reverse=True)

        # Perform dataset-specific preprocessing of the MRs
        self.mrs_as_lists = self.get_mrs(lowercase=self.convert_to_lowercase,
                                         slot_name_conversion=self.slot_name_conversion)

        # Lowercase utterances if needed
        self.utterances = self.get_utterances(lowercase=self.convert_to_lowercase)

        if self.num_slot_permutations > 0:
            self.augment_data_with_slot_permutations(self.num_slot_permutations)
            if self.partition != 'train':
                print('>> Warning: using slot permutation in a non-training partition of the dataset.')
                print()

        # Convert MRs back to strings
        add_separators = (self.slot_name_conversion == SlotNameConversionMode.VERBALIZE)
        self.mrs = [self.convert_mr_from_list_to_str(mr, add_separators=add_separators) for mr in self.mrs_as_lists]

        # DEBUG
        # print('>> MRs:\n{}'.format('\n'.join(self.mrs[:50])))
        # if isinstance(self.utterances[0], str):
        #     print('>> Utterances:\n{}'.format('\n'.join(self.utterances[:50])))
        # else:
        #     print('>> Utterances:\n{}'.format('\n'.join(['[' + '; '.join(utt) + ']' for utt in self.utterances[:10]])))

        if self.utterances:
            assert len(self.mrs) == len(self.utterances)

        # DEBUG
        # self.mrs = self.mrs[:10]
        # self.mrs_raw = self.mrs_raw[:10]
        # self.mrs_raw_as_lists = self.mrs_raw_as_lists[:10]
        # self.utterances = self.utterances[:10]
        # df_mrs = pd.DataFrame({'mr_orig': self.mrs_raw, 'mr': self.mrs})
        # df_mrs.to_csv(os.path.splitext(dataset_path)[0] + '_mrs.csv', index=False, encoding='utf-8-sig')

    def get_mrs(self, raw=False, lowercase=False, slot_name_conversion=None):
        if raw:
            mrs = self.mrs_raw
            if lowercase:
                print('Warning: raw MRs are returned with original letter case.')
        else:
            # Convert MRs to an intermediate format of lists of tuples, and cache the outputs
            if not self.mrs_raw_as_lists:
                self.mrs_raw_as_lists = [self.convert_mr_from_str_to_list(mr) for mr in self.mrs_raw]

            mrs = self.preprocess_mrs_in_intermediate_format(
                self.mrs_raw_as_lists, lowercase=lowercase, slot_name_conversion=slot_name_conversion)

        return mrs

    def get_utterances(self, lowercase=False):
        if lowercase:
            return self.lowercase_utterances(self.utterances)
        else:
            return self.utterances

    def augment_data_with_slot_permutations(self, num_permutations):
        """Augments the data with examples created by permuting content slots in the MR.

        Utterances are left unchanged in the new examples. Assumes data to be already preprocessed.
        """
        mrs_augm = []
        utterances_augm = []

        if any(sum(slot in ['da', 'intent'] for slot, _ in mr_as_list) for mr_as_list in self.mrs_as_lists) > 1:
            raise NotImplementedError('Slot permutation not supported for datasets with multiple DAs per MR')

        for mr_as_list, utt in zip(self.mrs_as_lists, self.utterances):
            mrs_augm.append(mr_as_list)
            da = mr_as_list[0] if mr_as_list[0][0] in ['da', 'intent'] else None
            content_slots = mr_as_list[1:] if da else mr_as_list

            for i in range(num_permutations):
                mr_augm = [da] if da else []
                mr_augm.extend(random.sample(content_slots, len(content_slots)))
                mrs_augm.append(mr_augm)

            utterances_augm.extend([utt] * (num_permutations + 1))

        self.mrs_as_lists = mrs_augm
        self.utterances = utterances_augm

    def create_reference_file_for_testing(self):
        """Creates a text file with groups of utterances corresponding to one MR separated by an empty line.

        A file with reference utterances in this format is necessary for the E2E NLG Challenge evaluation script.
        """
        out_file = os.path.join('eval', 'test_references_{}.txt'.format(self.name))

        with open(out_file, 'w', encoding='utf8') as f_out:
            if isinstance(self.utterances[0], list):
                for i in range(len(self.utterances)):
                    f_out.write('\n'.join(self.utterances[i]))
                    f_out.write('\n\n')
            elif isinstance(self.utterances[0], str):
                f_out.write('\n\n'.join(self.utterances))
                f_out.write('\n')

    def get_token_type_sequence(self, idx):
        """Pre-calculates token type sequences using dataset-specific mapping of slot names to salient tokens.

        Required when using slot name verbalization along with token type IDs in the GPT-2 model.
        """
        if not self.slot_name_conversion:
            token_type_seq = [self.get_single_word_slot_representation(slot[0]) for slot in self.mrs_raw_as_lists[idx]]
            if self.bos_token:
                token_type_seq.append(self.bos_token)

            return ' '.join(token_type_seq)
        else:
            return ''

    def calculate_da_based_weights(self):
        """Calculates MR weights inversely proportionate to their DA frequency in the dataset."""
        mrs = self.get_mrs(lowercase=True)
        da_list = [val for mr in mrs for slot, val in mr if slot in ['intent', 'da']]
        da_distr = Counter(da_list)
        weights = [len(mrs) / da_distr[da] for da in da_list]

        assert len(weights) == len(mrs), 'Some MRs appear to be missing DA type indication'

        print('>> DA distribution:')
        print('\n'.join([f'{key}: {val}' for key, val in da_distr.items()]))
        print()

        return weights

    @staticmethod
    def get_data_file_path(partition):
        raise NotImplementedError('method \'get_data_file_path\' must be defined by subclass')

    @staticmethod
    def read_data_from_dataframe(df_data, group_by_mr=False):
        # Extract the column names
        mr_col_name = df_data.columns[0]
        utt_col_name = df_data.columns[1] if df_data.shape[1] > 1 else None

        # Save the MRs and the utterances as lists (repeated MRs are collapsed for test data)
        if group_by_mr:
            # If utterances are present in the data
            if df_data.shape[1] > 1:
                # Group by MR, and aggregate utterances into lists
                df_grouped_by_mr = df_data.groupby(mr_col_name, sort=False)[utt_col_name].apply(list).reset_index()
                mrs = df_grouped_by_mr[mr_col_name].tolist()
                utterances = df_grouped_by_mr[utt_col_name].tolist()
            else:
                mrs = df_data[mr_col_name].tolist()
                utterances = []
        else:
            mrs = df_data[mr_col_name].tolist()
            if df_data.shape[1] > 1:
                utterances = df_data[utt_col_name].tolist()
            else:
                raise ValueError('Training and validation input data are expected to have two columns')

        return mrs, utterances

    @classmethod
    def convert_mr_from_str_to_list(cls, mr_as_str):
        """Converts an MR string to an intermediate format: a list of slot/value pairs.

        Any DA indications are first converted to the slot-and-value format. The original order of slots (and DAs) is
        preserved.
        """
        mr_as_str = cls.preprocess_da_in_mr(mr_as_str)
        mr_as_list = cls.parse_slot_and_value_pairs(mr_as_str)

        return mr_as_list

    @staticmethod
    def convert_mr_from_list_to_str(mr_as_list, add_separators=False):
        """Converts an MR from its intermediate format to a string intended as input to a model.

        The order of slots in the list is preserved. Special separators are added between slot/value pairs, as well as
        between a slot and its value, if add_separators is set to True.
        """
        slot_sep = ' | ' if add_separators else ' '
        val_sep = ' = ' if add_separators else ' '

        return slot_sep.join(['{0}{1}'.format(slot, val_sep + val if val else '') for slot, val in mr_as_list])

    @classmethod
    def preprocess_mrs_in_intermediate_format(cls, mrs, lowercase=False, slot_name_conversion=None):
        """Performs a series of preprocessing actions on MRs in the intermediate list-of-tuples format.

        Depending on the as_lists parameter, it returns the MRs either in the intermediate format or as strings.
        """
        # Preprocess slot names
        if slot_name_conversion:
            mrs = [cls.preprocess_slot_names_in_mr(mr, slot_name_conversion) for mr in mrs]

        # Preprocess slot values
        mrs = cls.preprocess_slot_values_in_mrs(mrs)

        if lowercase:
            # Convert slots and values to lowercase
            mrs = cls.lowercase_mrs(mrs)

        return mrs

    @classmethod
    def preprocess_mrs(cls, mrs, as_lists=False, lowercase=False, slot_name_conversion=None):
        """Performs dataset-specific preprocessing of the given MRs."""

        # Convert MRs to an intermediate format of lists of tuples
        mrs_as_lists = [cls.convert_mr_from_str_to_list(mr) for mr in mrs]

        # Perform dataset-specific preprocessing of the MRs, and convert them back to strings
        mrs_preprocessed = cls.preprocess_mrs_in_intermediate_format(
            mrs_as_lists, lowercase=lowercase, slot_name_conversion=slot_name_conversion)

        if as_lists:
            return mrs_preprocessed
        else:
            # Convert MRs to strings
            add_separators = (slot_name_conversion == SlotNameConversionMode.VERBALIZE)
            return [cls.convert_mr_from_list_to_str(mr, add_separators=add_separators) for mr in mrs_preprocessed]

    @classmethod
    def preprocess_da_in_mr(cls, mr):
        """Converts the DA type indication(s) in the MR (as a string) to the slot-and-value format."""

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
            match = regex.match(r'(\S+){0}(.*?){1}(?:{2}(\S+){0}(.*?){1})*$'.format(
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
    def parse_slot_and_value_pairs(cls, mr):
        """Extracts the sequence of slots and their corresponding values from the MR string as a list of tuples."""

        slots_and_values = []

        # Parse slots and their values
        match = regex.match(r'(.+?)(?:{0}(.*?){1})?(?:{2}(.+?)(?:{0}(.*?){1})?)*$'.format(
            re.escape(cls.delimiters['val_beg']), re.escape(cls.delimiters['val_end']),
            re.escape(cls.delimiters['slot_sep'])), mr)

        if not match:
            print(f'Warning: Unexpected format of the following MR:\n{mr}')
            return mr

        for i in range(1, len(match.groups()) + 1, 2):
            if match.group(i) is None:
                break

            for j in range(len(match.captures(i))):
                # Save the slot/value pair (note: value may be absent)
                slot = match.captures(i)[j].strip()
                value = match.captures(i + 1)[j].strip() if j < len(match.captures(i + 1)) else ''
                if value == '?':
                    value = ''
                slots_and_values.append((slot, value))

        return slots_and_values

    @classmethod
    def preprocess_slot_names_in_mr(cls, mr_as_list, slot_name_conversion):
        mr_processed = []

        for slot, value in mr_as_list:
            if slot_name_conversion == SlotNameConversionMode.SPECIAL_TOKENS:
                slot = cls.convert_slot_name_to_special_token(slot)
            elif slot_name_conversion == SlotNameConversionMode.VERBALIZE:
                if slot == 'da':
                    if '-' in value:
                        domain_slot_name = 'topic'
                        domain_value = cls.verbalize_domain_name(value)
                        mr_processed.append((domain_slot_name, domain_value))

                    slot = 'intent'
                    value = cls.verbalize_da_name(value)
                else:
                    slot = cls.verbalize_slot_name(slot)
            else:
                raise ValueError(f'Slot name conversion mode "{slot_name_conversion}" not recognized.')

            # Append a number to the slot name if a slot with the same name has already been encountered in the MR
            # slot_names = {slot_name for slot_name, value in mr_as_list}
            # if slot in slot_names:
            #     slot_new = slot
            #     slot_ctr = 1
            #     while slot_new in slot_names:
            #         slot_new = slot + str(slot_ctr)
            #         slot_ctr += 1
            #     slot = slot_new

            mr_processed.append((slot, value))

        return mr_processed

    @classmethod
    def preprocess_slot_values_in_mrs(cls, mrs):
        return mrs

    @classmethod
    def delexicalize_utterances(cls, mrs, utterances, lowercase=False):
        """Replaces slot values (obtained from the MRs) with placeholders in the corresponding utterances.

        Note that original slot names are used to check which slots should be delexicalized (dataset-specific), but slot
        values are preprocessed (e.g., MultiWOZ has slot values in MRs partially tokenized, and hence they need to be
        detokenized, such as "do n't" => "don't", in order to be correctly matched in the utterance).

        Slot values are matched in the utterance as whole words only, e.g., the value "action" will be matched in
        "full of action and adventure" or "action-packed game", but not in "reaction" or "actions".
        """
        delexicalized_utterances = []
        mrs_as_lists = cls.preprocess_slot_values_in_mrs([cls.convert_mr_from_str_to_list(mr) for mr in mrs])

        slots_to_delexicalize = cls.get_slots_to_delexicalize()
        simple_slots = slots_to_delexicalize.get('simple', [])
        list_slots = slots_to_delexicalize.get('list', [])

        for mr, utt in zip(mrs_as_lists, utterances):
            if lowercase:
                utt = utt.lower()

            for slot_name, slot_value in mr:
                # Ignore dialogue acts, and slots with an empty value
                if slot_name == 'da' or not slot_value:
                    continue

                if lowercase:
                    slot_value = slot_value.lower()
                placeholder = f'_|{slot_name}|_'

                if slot_name in simple_slots:
                    values_to_replace = [slot_value]
                elif slot_name in list_slots:
                    values_to_replace = [item.strip() for item in slot_value.split(',')]
                else:
                    values_to_replace = []

                for value in values_to_replace:
                    # Use regex with word boundary only when the slot value starts and ends with an alphanumeric
                    # character, otherwise the regex fails to match the value because most of the other symbols are
                    # considered a word boundary themselves)
                    if re.match(r'\w', value[0]) and re.match(r'\w', value[-1]):
                        utt = re.sub(r'\b{}\b'.format(re.escape(value)), placeholder, utt)
                    else:
                        utt = utt.replace(value, placeholder)

            delexicalized_utterances.append(utt)

        return delexicalized_utterances

    @staticmethod
    def lowercase_mrs(mrs):
        """Lowercases the given MRs."""
        if isinstance(mrs[0], list):
            return [[(slot.lower(), value.lower()) for slot, value in mr_as_list] for mr_as_list in mrs]
        elif isinstance(mrs[0], str):
            return [mr_as_str.lower() for mr_as_str in mrs]
        else:
            raise TypeError('MRs must be strings, or lists of slot-and-value tuples.')

    @staticmethod
    def lowercase_utterances(utterances):
        """Lowercases the given utterances."""
        if isinstance(utterances[0], str):
            return [utt.lower() for utt in utterances]
        elif isinstance(utterances[0], list):
            return [[utt.lower() for utt in utt_list] for utt_list in utterances]
        else:
            raise TypeError('Utterances must be strings, or lists of strings.')

    @staticmethod
    def sort_data(mrs, utterances, reverse=False):
        """Sorts the data (MRs and utterances in separate lists) by the approximate MR length in words.

        MRs are split into words just by whitespace characters, rather than using tokenization.
        """
        if utterances:
            mrs, utterances = zip(*sorted(zip(mrs, utterances), key=lambda x: len(x[0].split()), reverse=reverse))
            mrs, utterances = list(mrs), list(utterances)
        else:
            mrs = sorted(mrs, key=lambda x: len(x.split), reverse=reverse)

        # DEBUG
        # print('>> MR length samples:', [len(mrs[idx].split()) for idx in range(0, len(mrs), len(mrs) // 50)])
        # print()

        return mrs, utterances

    @classmethod
    def identify_boolean_slots(cls, ontology, additional_bool_values=None):
        """Extracts the set of all Boolean slots in the dataset's ontology, inferred from their possible values.

        To specify additional Boolean values, a nested list of strings can be passed as the `additional_bool_values`
        parameter. Note that slot values are lowercased before being matched with the specified Boolean values.
        """
        bool_slots = set()

        # Predefined possible Boolean and none-values
        bool_value_groups = [['yes', 'no'], ['true', 'false']]
        none_values = ['', '?', 'none']

        # Add any additional Boolean values passed to the method
        if additional_bool_values and isinstance(additional_bool_values, list):
            for bool_values_to_add in additional_bool_values:
                if isinstance(bool_values_to_add, list):
                    bool_value_groups.append(bool_values_to_add)

        # Iterate over all slots in the ontology and record those whose all values are Boolean or none-values
        for slot_name, values in ontology.items():
            for bool_values in bool_value_groups:
                bool_and_none_values = bool_values + none_values
                if all(value.lower() in bool_and_none_values for value in values):
                    bool_slots.add(slot_name)

        print('>> Boolean slots identified in the dataset:')
        print(', '.join(sorted(bool_slots)))
        print()

        return bool_slots

    @classmethod
    def get_special_tokens(cls, slot_name_conversion=None):
        slot_tokens = set()

        if slot_name_conversion == SlotNameConversionMode.SPECIAL_TOKENS:
            train_set_path = cls.get_data_file_path('train')
            df_data = pd.read_csv(train_set_path, header=0, encoding='utf8')

            for mr_as_str in df_data[df_data.columns[0]]:
                mr_as_list = cls.convert_mr_from_str_to_list(mr_as_str)
                for slot, value in mr_as_list:
                    slot = cls.convert_slot_name_to_special_token(slot)
                    slot_tokens.add(slot)

        return sorted(list(slot_tokens))

    @staticmethod
    def convert_slot_name_to_special_token(slot_name):
        """Converts a slot name to a special token."""
        return '<|{}|>'.format(slot_name.replace(' ', '').lower())

    @staticmethod
    def verbalize_da_name(da_name):
        raise NotImplementedError('method \'verbalize_da_name\' must be defined by subclass')

    @staticmethod
    def verbalize_domain_name(da_name):
        raise NotImplementedError('method \'verbalize_domain_name\' must be defined by subclass')

    @staticmethod
    def verbalize_slot_name(slot_name):
        raise NotImplementedError('method \'verbalize_slot_name\' must be defined by subclass')

    @staticmethod
    def get_single_word_slot_representation(slot_name):
        raise NotImplementedError('method \'get_single_word_slot_representation\' must be defined by subclass')

    @staticmethod
    def get_slots_to_delexicalize():
        """Returns a set of slot names (in their original form) for which delexicalization in utterances makes sense."""
        raise NotImplementedError('method \'get_slots_to_delexicalize\' must be defined by subclass')
