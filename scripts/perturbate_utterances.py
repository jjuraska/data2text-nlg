import pandas as pd
import random

from data_loader import MRToTextDataset
from ontology import DatasetOntologyBuilder
from typing import List, Optional, Sequence, Set, Tuple, Type, Union


SPECIAL_SLOT_VALUE_SET = {'', 'none', '?'}


class PerturbationMode(object):
    DELETE = 'delete'
    DUPLICATE = 'duplicate'
    INSERT = 'insert'
    NEGATE_BOOL = 'negate_bool'
    SHUFFLE = 'shuffle'
    SUBSTITUTE = 'substitute'


class SlotNameMode(object):
    SINGLE_WORD = 'single_word'
    VERBALIZED = 'verbalized'


class PseudoUtteranceGenerator(object):
    def __init__(self, dataset_class: Type[MRToTextDataset], partition: Optional[str] = None,
                 file_path: Optional[str] = None) -> None:
        self.dataset_class = dataset_class

        if file_path:
            self.mrs, self.utterances = self.load_data_from_file(file_path)
        elif partition:
            self.mrs, self.utterances = self.load_dataset_partition(partition)
        else:
            raise ValueError('Either a partition name or a file path must be provided when creating a '
                             '`PseudoUtteranceGenerator` object.')

        ontology_builder = DatasetOntologyBuilder(self.dataset_class, load_from_file=True, ignore_aux_slots=True)
        self.ontology = ontology_builder.ontology
        self.bool_slots = self.dataset_class.identify_boolean_slots(self.ontology)

    def load_dataset_partition(self, partition: str) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
        if partition not in {'train', 'valid', 'test'}:
            raise ValueError(f'Partition "{partition}" not recognized.')

        df_data = pd.read_csv(self.dataset_class.get_data_file_path(partition), header=0, encoding='utf-8')
        mrs, references = self.dataset_class.read_data_from_dataframe(df_data, group_by_mr=False)
        mrs_as_lists = [self.dataset_class.convert_mr_from_str_to_list(mr) for mr in mrs]

        return mrs_as_lists, references

    def load_data_from_file(self, file_path: str) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
        df_data = pd.read_csv(file_path, header=0, encoding='utf-8')
        mrs, references = self.dataset_class.read_data_from_dataframe(df_data, group_by_mr=False)
        mrs_as_lists = [self.dataset_class.convert_mr_from_str_to_list(mr) for mr in mrs]

        return mrs_as_lists, references

    def generate_pseudo_utterances(self, separator: str = ' ', include_slot_names: bool = False,
                                   bool_slot_name_mode: str = SlotNameMode.VERBALIZED, lowercase: bool = False,
                                   perturbation: Optional[str] = None, perturbation_n: int = 1) -> List[str]:
        pseudo_utterances = []
        mrs = [self._remove_da_slot_from_mr(mr) for mr in self.mrs]

        if perturbation:
            mrs = self.perturbate_mrs(mrs, perturbation, num_slots_per_mr=perturbation_n)

        for mr in mrs:
            content = []

            for slot_name, slot_value in mr:
                is_bool_slot = slot_name in self.bool_slots
                if is_bool_slot or not slot_value:
                    if bool_slot_name_mode == SlotNameMode.SINGLE_WORD:
                        slot_name_verbalized = self.dataset_class.get_single_word_slot_representation(slot_name)
                    elif bool_slot_name_mode == SlotNameMode.VERBALIZED:
                        slot_name_verbalized = self.dataset_class.verbalize_slot_name(slot_name)
                    else:
                        raise ValueError(f'Slot name mode "{bool_slot_name_mode}" not recognized.')

                    if is_bool_slot and slot_value in {'no', 'false'}:
                        slot_name_verbalized = 'not ' + slot_name_verbalized
                    content.append(slot_name_verbalized)
                elif include_slot_names:
                    slot_name_verbalized = self.dataset_class.verbalize_slot_name(slot_name)
                    content.append(f'{slot_name_verbalized} {slot_value}')
                else:
                    content.append(slot_value)

            pseudo_utt = separator.join(content)
            pseudo_utterances.append(pseudo_utt.lower() if lowercase else pseudo_utt)

        return pseudo_utterances

    def generate_mrs(self, exclude_da: bool = False, lowercase: bool = False, perturbation: Optional[str] = None,
                     perturbation_n: int = 1, return_as_lists: bool = False) -> List[str]:
        mrs_pert = []

        if exclude_da:
            das = [None] * len(self.mrs)
        else:
            das = [self._identify_da_slot_in_mr(mr) for mr in self.mrs]
        mrs = [self._remove_da_slot_from_mr(mr) for mr in self.mrs]

        if perturbation:
            mrs = self.perturbate_mrs(mrs, perturbation, num_slots_per_mr=perturbation_n)

        if return_as_lists:
            return mrs
        else:
            for mr, da in zip(mrs, das):
                mr_pert = self._reconstruct_mr(mr, da)
                mrs_pert.append(mr_pert.lower() if lowercase else mr_pert)

            return mrs_pert

    def _identify_da_slot_in_mr(self, mr_as_list: Sequence[Tuple[str, str]]) -> Optional[str]:
        for slot_name, slot_value in mr_as_list:
            if slot_name == 'da':
                return slot_value

        return None

    def _remove_da_slot_from_mr(self, mr_as_list: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return [(slot_name, slot_value) for slot_name, slot_value in mr_as_list if slot_name != 'da']

    def _reconstruct_mr(self, mr_as_list: Sequence[Tuple[str, str]], da: Optional[str] = None) -> str:
        slot_strings = []

        for slot_name, slot_val in mr_as_list:
            slot_strings.append(f'{slot_name}[{slot_val}]')

        slots_and_values_str = ', '.join(slot_strings)
        return f'{da}({slots_and_values_str})' if da else slots_and_values_str

    def perturbate_mrs(self, mrs: Sequence[List[Tuple[str, str]]], mode: str,
                       num_slots_per_mr: int = 1) -> List[List[Tuple[str, str]]]:
        mrs_new = []

        for mr in mrs:
            if mode == PerturbationMode.SHUFFLE:
                mr_new = self.shuffle_slots_in_mr(mr)
                mrs_new.append(mr_new)
            else:
                idxs_to_modify = self._choose_slot_indices_to_modify(mr, mode, n=num_slots_per_mr)
                if idxs_to_modify:
                    if mode == PerturbationMode.DELETE:
                        mr_new = self.delete_n_slots_in_mr(mr, idxs_to_modify)
                    elif mode == PerturbationMode.SUBSTITUTE:
                        mr_new = self.substitute_n_slot_values_in_mr(mr, idxs_to_modify)
                    elif mode == PerturbationMode.DUPLICATE:
                        mr_new = self.duplicate_n_slots_in_mr(mr, idxs_to_modify)
                    elif mode == PerturbationMode.INSERT:
                        mr_new = self.insert_n_slots_in_mr(mr, idxs_to_modify)
                    elif mode == PerturbationMode.NEGATE_BOOL:
                        mr_new = self.negate_n_bool_slot_values_in_mr(mr, idxs_to_modify)
                    else:
                        raise ValueError(f'Perturbation "{mode}" not recognized')
                    mrs_new.append(mr_new)
                else:
                    mrs_new.append(mr)

        return mrs_new

    def _choose_slot_indices_to_modify(self, mr: List[Tuple[str, str]], mode: str,
                                       n: int) -> Union[Set[int], List[int]]:
        if n <= 0:
            return set()

        if mode == PerturbationMode.INSERT:
            # Choose indices for slots not present in the MR, possibly repeating indices if MR has fewer slots than n
            num_absent_slots = len(self.ontology.keys()) - len(set(slot_name for slot_name, _ in mr))
            num_slots_to_modify = min(n, num_absent_slots)
            if num_slots_to_modify > len(mr):
                idx_choices = range(len(mr)) if len(mr) > 0 else [0]
                return random.choices(idx_choices, k=num_slots_to_modify)       # Sample with repetition
            else:
                return random.sample(range(len(mr)), num_slots_to_modify)       # Sample without repetition
        elif mode == PerturbationMode.NEGATE_BOOL:
            # Choose up to n slots from among the MR's Boolean slots
            bool_slot_idxs = [i for i, (slot_name, slot_value) in enumerate(mr) if slot_name in self.bool_slots]
            num_slots_to_modify = min(n, len(bool_slot_idxs))
            return set(random.sample(bool_slot_idxs, num_slots_to_modify))
        else:
            # Choose up to n slots in the MR, but keep at least one slot unmodified in case of the "delete" perturbation
            if mode == PerturbationMode.DELETE:
                num_slots_to_modify = min(n, len(mr) - 1)
            else:
                num_slots_to_modify = min(n, len(mr))
            return set(random.sample(range(len(mr)), num_slots_to_modify))

    def duplicate_n_slots_in_mr(self, mr: List[Tuple[str, str]],
                                idxs_to_duplicate: Set[int]) -> List[Tuple[str, str]]:
        return mr + [mr[idx] for idx in idxs_to_duplicate]

    def insert_n_slots_in_mr(self, mr: List[Tuple[str, str]], idxs_to_insert: List[int]) -> List[Tuple[str, str]]:
        mr_new = mr
        slots_present = set(slot_name for slot_name, _ in mr)
        slots_absent = set(self.ontology.keys()) - slots_present

        for idx in sorted(idxs_to_insert, reverse=True):
            slot_to_insert = random.choice(list(slots_absent))
            value_to_insert = random.choice(list(self.ontology[slot_to_insert]))
            mr_new.insert(idx, (slot_to_insert, value_to_insert))
            slots_present.add(slot_to_insert)
            slots_absent.remove(slot_to_insert)

        return mr_new

    def substitute_n_slot_values_in_mr(self, mr: List[Tuple[str, str]],
                                       idxs_to_substitute: Set[int]) -> List[Tuple[str, str]]:
        mr_new = []

        for idx, (slot_name, slot_value) in enumerate(mr):
            if idx in idxs_to_substitute:
                # In case of an empty/special value, replace the slot, otherwise replace its value only
                if slot_value in SPECIAL_SLOT_VALUE_SET:
                    alt_slot_candidates = set(self.ontology.keys()) - {slot_name}
                    alt_slot = random.choice(list(alt_slot_candidates))
                    mr_new.append((alt_slot, slot_value))
                else:
                    alt_value_candidates = set(self.ontology[slot_name]) - {slot_value} - SPECIAL_SLOT_VALUE_SET
                    alt_value = random.choice(list(alt_value_candidates))
                    mr_new.append((slot_name, alt_value))
            else:
                mr_new.append((slot_name, slot_value))

        return mr_new

    def delete_n_slots_in_mr(self, mr: List[Tuple[str, str]], idxs_to_delete: Set[int]) -> List[Tuple[str, str]]:
        return [mr[idx] for idx in range(len(mr)) if idx not in idxs_to_delete]

    def negate_n_bool_slot_values_in_mr(self, mr: List[Tuple[str, str]],
                                        idxs_to_negate: Set[int]) -> List[Tuple[str, str]]:
        mr_new = []

        for idx, (slot_name, slot_value) in enumerate(mr):
            if idx in idxs_to_negate and slot_value not in SPECIAL_SLOT_VALUE_SET:
                alt_value_candidates = set(self.ontology[slot_name]) - {slot_value} - SPECIAL_SLOT_VALUE_SET
                alt_value = random.choice(list(alt_value_candidates))
                mr_new.append((slot_name, alt_value))
            else:
                mr_new.append((slot_name, slot_value))

        return mr_new

    def shuffle_slots_in_mr(self, mr: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return random.sample(mr, len(mr))
