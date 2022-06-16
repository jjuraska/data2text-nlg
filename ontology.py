from collections import defaultdict, OrderedDict
import json
import os
import pandas as pd
from typing import Any, Dict, Union

from constants import SlotNameConversionMode


class DatasetOntologyBuilder(object):
    def __init__(self, dataset_class: Any, load_from_file: Union[bool, str] = False,
                 preprocess_slot_names: bool = False, ignore_aux_slots: bool = False) -> None:
        self._dataset_class = dataset_class
        self._preprocess_slot_names = preprocess_slot_names
        self._ignore_aux_slots = ignore_aux_slots
        self._aux_slots = ['da', 'intent', 'topic']
        self._ontology = defaultdict(set)

        if load_from_file:
            file_path = load_from_file if isinstance(load_from_file, str) else None
            try:
                self._load(file_path=file_path)
            except FileNotFoundError:
                if file_path:
                    print(f'>> Warning: File "{file_path}" not found, building a new ontology.')
                else:
                    print('>> Warning: Default ontology has not been created yet, building a new one now.')
                self._build()
                self.export(file_path=file_path)
        else:
            self._build()

    @property
    def ontology(self) -> Dict:
        return self._ontology

    def _build(self) -> None:
        """Creates an ontology of the dataset, listing all possible values for each slot.

        The ontology is created based on the training set only.
        """
        ontology_dict = defaultdict(set)

        train_set_path = self._dataset_class.get_data_file_path('train')
        df_data = pd.read_csv(train_set_path, header=0, encoding='utf-8')

        for mr_as_str in df_data[df_data.columns[0]]:
            mr_as_list = self._dataset_class.convert_mr_from_str_to_list(mr_as_str)
            if self._preprocess_slot_names:
                mr_as_list = self._dataset_class.preprocess_slot_names_in_mr(
                    mr_as_list, SlotNameConversionMode.VERBALIZE)

            for slot, value in mr_as_list:
                if self._ignore_aux_slots and slot in self._aux_slots:
                    continue
                ontology_dict[slot].add(value)

        self._ontology = ontology_dict

    def _load(self, file_path: str = None) -> None:
        """Loads ontology from a file, if path is provided, otherwise from the default path specified by the dataset."""
        if file_path:
            # Check the input file's extension
            file_ext = os.path.splitext(file_path)[1]
            if file_ext != '.json':
                print('Warning: The ontology is expected to be in the JSON format, but the provided input file name\'s'
                      ' extension is \"{file_ext}\"')
        else:
            file_path = self.get_default_path()

        # Load from a JSON file
        with open(file_path, 'r', encoding='utf-8') as f_in:
            ontology_dict = json.load(f_in)

        if self._ignore_aux_slots:
            for key in self._aux_slots:
                ontology_dict.pop(key, None)

        self._ontology = ontology_dict

        print(f'>> Dataset ontology loaded from "{file_path}"')

    def export(self, file_path: str = None) -> None:
        """Saves the ontology to the specified file, if provided, otherwise to a JSON file in the dataset folder."""
        # Sort both slots and their value sets alphabetically
        ontology_sorted = OrderedDict(
            {slot: sorted(value_set) for slot, value_set in sorted(self._ontology.items(), key=lambda x: x[0])})

        if file_path:
            # Check the output file's extension
            file_ext = os.path.splitext(file_path)[1]
            if file_ext != '.json':
                print('Warning: The ontology will be exported as a JSON file, but the provided output file name\'s'
                      ' extension is \"{file_ext}\"')
        else:
            file_path = self.get_default_path()

        # Save to a JSON file
        with open(file_path, 'w', encoding='utf-8') as f_out:
            json.dump(ontology_sorted, f_out, indent=4, ensure_ascii=False)

        print(f'>> Dataset ontology exported to "{file_path}"')

    def get_default_path(self) -> str:
        """Returns a file path pointing to a JSON ontology file directly in the dataset folder."""
        dataset_dir = os.path.dirname(self._dataset_class.get_data_file_path('train'))
        file_name = 'ontology_preprocessed.json' if self._preprocess_slot_names else 'ontology.json'

        return os.path.join(dataset_dir, file_name)
