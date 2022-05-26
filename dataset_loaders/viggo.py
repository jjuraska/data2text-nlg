import os
import re

from data_loader import MRToTextDataset


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
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path

    @staticmethod
    def verbalize_da_name(da_name):
        return da_name.replace('_', ' ')

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'esrb': 'ESRB rating',
            'exp_release_date': 'expected release date',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.replace('_', ' ')
            for tok in ['linux', 'mac', 'steam']:
                slot_name_verbalized = re.sub(r'\b{}\b'.format(re.escape(tok)), tok.capitalize(), slot_name_verbalized)

        return slot_name_verbalized

    @staticmethod
    def get_single_word_slot_representation(slot_name):
        single_word_slot_repr = {
            'available_on_steam': 'Steam',
            'da': 'intent',
            # 'esrb': 'content',
            'esrb': 'ESRB',
            'exp_release_date': 'expected',
            'has_linux_release': 'Linux',
            'has_mac_release': 'Mac',
            'has_multiplayer': 'multiplayer',
            'player_perspective': 'perspective',
            'release_year': 'year',
            # 'specifier': 'specify',
            'specifier': 'specifier',
        }

        return single_word_slot_repr.get(slot_name, slot_name)

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {'developer', 'esrb', 'exp_release_date', 'name', 'release_year', 'specifier'},
            'list': {'genres', 'platforms', 'player_perspective'}
        }


class ViggoWithE2EDataset(ViggoDataset):
    """The ViGGO dataset with the training set merged with that of the E2E dataset."""
    name = 'video_game'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_with_e2e.csv')
        else:
            dataset_path = super(ViggoWithE2EDataset, ViggoWithE2EDataset).get_data_file_path(partition)

        return dataset_path


class Viggo20PercentDataset(ViggoDataset):
    """A 20% sample of the ViGGO dataset."""
    name = 'video_game_20_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.2.csv')
        else:
            dataset_path = super(Viggo20PercentDataset, Viggo20PercentDataset).get_data_file_path(partition)

        return dataset_path


class Viggo10PercentDataset(ViggoDataset):
    """A 10% sample of the ViGGO dataset."""
    name = 'video_game_10_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.1.csv')
        else:
            dataset_path = super(Viggo10PercentDataset, Viggo10PercentDataset).get_data_file_path(partition)

        return dataset_path


class Viggo5PercentDataset(ViggoDataset):
    """A 5% sample of the ViGGO dataset."""
    name = 'video_game_5_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.05.csv')
        else:
            dataset_path = super(Viggo5PercentDataset, Viggo5PercentDataset).get_data_file_path(partition)

        return dataset_path


class Viggo2PercentDataset(ViggoDataset):
    """A 2% sample of the ViGGO dataset."""
    name = 'video_game_2_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.02.csv')
        else:
            dataset_path = super(Viggo2PercentDataset, Viggo2PercentDataset).get_data_file_path(partition)

        return dataset_path


class Viggo1PercentDataset(ViggoDataset):
    """A 1% sample of the ViGGO dataset."""
    name = 'video_game_1_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.01.csv')
        else:
            dataset_path = super(Viggo1PercentDataset, Viggo1PercentDataset).get_data_file_path(partition)

        return dataset_path
