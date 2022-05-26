import os

from data_loader import MRToTextDataset


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
        dataset_dir = os.path.join('data', 'rest_e2e')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devset.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'testset.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'trainset.csv')

        return dataset_path

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'eatType': 'eatery type',
            'familyFriendly': 'family-friendly',
            'priceRange': 'price range',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name

        return slot_name_verbalized

    @staticmethod
    def get_single_word_slot_representation(slot_name):
        single_word_slot_repr = {
            'customer rating': 'rating',
            'eatType': 'type',
            'familyFriendly': 'family',
            'priceRange': 'price',
        }

        return single_word_slot_repr.get(slot_name, slot_name)

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {'area', 'eatType', 'food', 'name', 'near'}
        }


class E2ECleanedDataset(E2EDataset):
    """A cleaned version of the E2E dataset in the restaurant domain."""
    name = 'rest_e2e_cleaned'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'rest_e2e_cleaned')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devel-fixed.no-ol.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test-fixed.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train-fixed.no-ol.csv')

        return dataset_path
