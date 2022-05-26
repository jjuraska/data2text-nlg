import os
import re
from sacremoses import MosesDetokenizer

from data_loader import MRToTextDataset


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
        dataset_dir = os.path.join('data', 'multiwoz')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path

    @staticmethod
    def verbalize_da_name(da_name):
        das_to_override = {
            'reqmore': 'request more',
        }

        # Parse the DA name from the domain+DA indication
        da_name = re.match(r'(?:\w+)-(\w+)', da_name).group(1)

        if da_name in das_to_override:
            da_name_verbalized = das_to_override[da_name]
        else:
            if re.match(r'^[A-Z]', da_name):
                da_name_verbalized = ' '.join([tok.lower() for tok in re.findall('[A-Z][^A-Z]*', da_name)])
            else:
                da_name_verbalized = da_name.lower()

        return da_name_verbalized

    @staticmethod
    def verbalize_domain_name(domain_name):
        # Parse the domain name from the domain+DA indication
        domain_name = re.match(r'(\w+)-(?:\w+)', domain_name).group(1)

        return domain_name.lower()

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'Addr': 'address',
            'Dest': 'destination',
            'Id': 'ID',
            'Internet': 'Internet',
            'People': 'number of people',
            'Phone': 'phone number',
            'Post': 'postcode',
            'Price': 'price range',
            'Ref': 'reference number',
            'Stars': 'rating',
            'Stay': 'length of stay',
            'Ticket': 'ticket price',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.lower()

        return slot_name_verbalized

    @classmethod
    def preprocess_slot_values_in_mrs(cls, mrs):
        # Detokenize slot values
        detokenizer = MosesDetokenizer()
        return [[(slot, cls.detokenize_slot_value(value, detokenizer)) for slot, value in mr] for mr in mrs]

    @staticmethod
    def detokenize_slot_value(value, detokenizer):
        value_detok = re.sub(r'(?<=\s): (?=\d)', ':', value)
        value_detok = detokenizer.detokenize(value_detok.split())
        value_detok = value_detok.replace(" - ", "-").replace(" n't", "n't").replace("I 'm", "I'm")

        return value_detok

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {
                'Addr', 'Area', 'Arrive', 'Car', 'Choice', 'Day', 'Depart', 'Department', 'Dest', 'Fee', 'Food', 'Id',
                'Leave', 'Name', 'People', 'Phone', 'Post', 'Ref', 'Stay', 'Ticket', 'Time', 'Type'
            }
        }
