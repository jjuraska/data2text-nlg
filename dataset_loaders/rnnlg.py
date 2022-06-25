import os
import re

from data_loader import MRToTextDataset


class LaptopDataset(MRToTextDataset):
    """An MR-to-text dataset in the laptop domain (provided as part of the RNNLG benchmark toolkit)."""
    name = 'laptop_rnnlg'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': None,
        'slot_sep': ';',
        'val_beg': '=',
        'val_end': ''
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'laptop_rnnlg')
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

        # Remove leading question mark present in some DA names
        if da_name.startswith('?'):
            da_name = re.sub(r'^\?', '', da_name)

        if da_name in das_to_override:
            return das_to_override[da_name]
        else:
            return da_name.replace('_', ' ')

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'batteryrating': 'battery rating',
            'driverange': 'drive range',
            'isforbusinesscomputing': 'is for business computing',
            'pricerange': 'price range',
            'weight range': 'weight range',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name

        return slot_name_verbalized

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {
                'battery', 'count', 'dimension', 'drive', 'family', 'memory', 'name', 'platform', 'price',
                'processor', 'type', 'utility', 'warranty', 'weight'
            },
            'list': {'design'}
        }


class TVDataset(MRToTextDataset):
    """An MR-to-text dataset in the TV domain (provided as part of the RNNLG benchmark toolkit)."""
    name = 'tv_rnnlg'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': None,
        'slot_sep': ';',
        'val_beg': '=',
        'val_end': ''
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'tv_rnnlg')
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

        # Remove leading question mark present in some DA names
        if da_name.startswith('?'):
            da_name = re.sub(r'^\?', '', da_name)

        if da_name in das_to_override:
            return das_to_override[da_name]
        else:
            return da_name.replace('_', ' ')

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'ecorating': 'eco rating',
            'hasusbport': 'has USB port',
            'hdmiport': 'HDMI port',
            'powerconsumption': 'power consumption',
            'pricerange': 'price range',
            'screensize': 'screen size',
            'screensizerange': 'screen size range',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name

        return slot_name_verbalized

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {
                'accessories', 'audio', 'count', 'ecorating', 'family', 'hdmiport', 'name', 'powerconsumption', 'price',
                'resolution', 'screensize', 'type'
            },
            'list': {'color'}
        }
