from collections import Counter, OrderedDict
import json
import os
import pandas as pd

from dataset_loaders.e2e import E2EDataset, E2ECleanedDataset
from dataset_loaders.multiwoz import MultiWOZDataset
from dataset_loaders.viggo import ViggoDataset
from slot_aligner.slot_alignment import count_errors, find_alignment


def align_slots(data_dir, filename, dataset_class, serialize_pos_info=False):
    """Finds and records the position of each slot's mention in the corresponding utterance.

    The position is indicated as the index of the first character of the slot mention phrase within the utterance. When
    the phrase comprises non-contiguous words in the utterance, the position is typically that of the salient term in
    the phrase.

    Note that if a slot is mentioned in the corresponding utterance multiple times, only its last mention is recorded.
    """
    alignments = []

    # Load MRs and corresponding utterances
    df_data = pd.read_csv(os.path.join(data_dir, filename), header=0)
    mrs_raw = df_data.iloc[:, 0].to_list()
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, convert_slot_names=True)
    utterances = df_data.iloc[:, 1].to_list()

    for mr_as_list, utt in zip(mrs_processed, utterances):
        # Determine the positions of all slot mentions in the utterance
        slot_mention_positions = find_alignment(utt, mr_as_list, dataset_class.name)
        if serialize_pos_info:
            alignments.append(json.dumps([[pos, slot] for slot, _, pos in slot_mention_positions]))
        else:
            alignments.append(' '.join([f'({pos}: {slot})' for slot, _, pos in slot_mention_positions]))

    # Save the MRs and utterances along with the positional information about slot mentions to a new CSV file
    df_data['alignment'] = alignments
    out_file_path = os.path.splitext(os.path.join(data_dir, filename))[0] + ' [with alignment].csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')


def score_slot_realizations(data_dir, predictions_file, dataset_class, slot_level=False, verbose=False):
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""

    error_counts = []
    incorrect_slots = []
    duplicate_slots = []
    total_content_slots = 0

    # Load MRs and corresponding utterances
    df_data = pd.read_csv(os.path.join(data_dir, predictions_file), header=0)
    mrs_raw = df_data.iloc[:, 0].to_list()
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, convert_slot_names=True)
    utterances = df_data.iloc[:, 1].fillna('').to_list()

    for mr_as_list, utt in zip(mrs_processed, utterances):
        # Count the missing and hallucinated slots in the utterance
        num_errors, cur_incorrect_slots, cur_duplicate_slots, num_content_slots = count_errors(
            utt, mr_as_list, dataset_class.name, verbose=verbose)
        error_counts.append(num_errors)
        incorrect_slots.append(', '.join(cur_incorrect_slots))
        duplicate_slots.append(', '.join(cur_duplicate_slots))
        total_content_slots += num_content_slots

    # Save the MRs and utterances along with their slot error indications to a new CSV file
    df_data['errors'] = error_counts
    df_data['incorrect'] = incorrect_slots
    df_data['duplicate'] = duplicate_slots
    out_file_path = os.path.splitext(os.path.join(data_dir, predictions_file))[0] + ' [errors].csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    # Calculate the slot-level or utterance-level SER
    if slot_level:
        ser = sum(error_counts) / total_content_slots
    else:
        ser = sum([num_errs > 0 for num_errs in error_counts]) / len(utterances)

    # Print the SER
    if verbose:
        print(f'>> Slot error rate: {round(100 * ser, 2)}%')
    else:
        print(f'{round(100 * ser, 2)}%')

    return ser


def score_emphasis(dataset, filename):
    """Determines how many of the indicated emphasis instances are realized in the utterance."""

    emph_missed = []
    emph_total = []

    print('Analyzing emphasis realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        expect_emph = False
        emph_slots = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be emphasized
            if slot == config.EMPH_TOKEN:
                expect_emph = True
            else:
                mr_dict[slot] = value
                if expect_emph:
                    emph_slots.add(slot)
                    expect_emph = False

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        emph_total.append(len(emph_slots))

        # Check how many emphasized slots were not realized before the name-slot
        for pos, slot, _ in alignment:
            # DEBUG PRINT
            # print(alignment)
            # print(emph_slots)
            # print()

            if slot == 'name':
                break

            if slot in emph_slots:
                emph_slots.remove(slot)

        emph_missed.append(len(emph_slots))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed emphasis', 'total emphasis'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed emphasis'] = emph_missed
    new_df['total emphasis'] = emph_total

    filename_out = os.path.splitext(filename)[0] + ' [emphasis eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_contrast(dataset, filename):
    """Determines whether the indicated contrast relation is correctly realized in the utterance."""

    contrast_connectors = ['but', 'however', 'yet']
    contrast_missed = []
    contrast_incorrectness = []
    contrast_total = []

    print('Analyzing contrast realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        contrast_found = False
        contrast_correct = False
        contrast_slots = []
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be contrasted
            if slot in [config.CONTRAST_TOKEN, config.CONCESSION_TOKEN]:
                contrast_slots.extend(value.split())
            else:
                mr_dict[slot] = value

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        contrast_total.append(1 if len(contrast_slots) > 0 else 0)

        if len(contrast_slots) > 0:
            for contrast_conn in contrast_connectors:
                contrast_pos = utterances[i].find(contrast_conn)
                if contrast_pos < 0:
                    continue

                slot_left_pos = -1
                slot_right_pos = -1
                dist = 0

                contrast_found = True

                # Check whether the correct pair of slots was contrasted
                for pos, slot, _ in alignment:
                    # DEBUG PRINT
                    # print(alignment)
                    # print(contrast_slots)
                    # print()

                    if slot_left_pos > -1:
                        dist += 1

                    if slot in contrast_slots:
                        if slot_left_pos == -1:
                            slot_left_pos = pos
                        else:
                            slot_right_pos = pos
                            break

                if slot_left_pos > -1 and slot_right_pos > -1:
                    if slot_left_pos < contrast_pos < slot_right_pos and dist <= 2:
                        contrast_correct = True
                        break
        else:
            contrast_found = True
            contrast_correct = True

        contrast_missed.append(0 if contrast_found else 1)
        contrast_incorrectness.append(0 if contrast_correct else 1)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed contrast', 'incorrect contrast', 'total contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed contrast'] = contrast_missed
    new_df['incorrect contrast'] = contrast_incorrectness
    new_df['total contrast'] = contrast_total

    filename_out = os.path.splitext(filename)[0] + ' [contrast eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def analyze_contrast_relations(dataset, filename):
    """Identifies the slots involved in a contrast relation."""

    contrast_connectors = ['but', 'however', 'yet']
    slots_before = []
    slots_after = []

    print('Analyzing contrast relations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]

    for mr, utt in zip(mrs, utterances_orig):
        mr_dict = OrderedDict()
        mr_list_augm = []

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            mr_list_augm.append((slot, value_orig))

        # Find the slot alignment
        alignment = find_alignment(utt, mr_dict)

        slot_before = None
        slot_after = None

        for contrast_conn in contrast_connectors:
            contrast_pos = utt.find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                slot_after = None

                for pos, slot, value in alignment:
                    slot_before = slot_after
                    slot_after = slot

                    if pos > contrast_pos:
                        break

                break

        slots_before.append(slot_before if slot_before is not None else '')
        slots_after.append(slot_after if slot_after is not None else '')

    # Calculate the frequency distribution of slots involved in a contrast relation
    contrast_slot_cnt = Counter()
    contrast_slot_cnt.update(slots_before + slots_after)
    del contrast_slot_cnt['']
    print('\n---- Slot distribution in contrast relations ----\n')
    print('\n'.join(slot + ': ' + str(freq) for slot, freq in contrast_slot_cnt.most_common()))

    # Calculate the frequency distribution of slot pairs involved in a contrast relation
    contrast_slot_cnt = Counter()
    slot_pairs = [tuple(sorted(slot_pair)) for slot_pair in zip(slots_before, slots_after) if slot_pair != ('', '')]
    contrast_slot_cnt.update(slot_pairs)
    print('\n---- Slot pair distribution in contrast relations ----\n')
    print('\n'.join(slot_pair[0] + ', ' + slot_pair[1] + ': ' + str(freq) for slot_pair, freq in contrast_slot_cnt.most_common()))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'slot before contrast', 'slot after contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['slot before contrast'] = slots_before
    new_df['slot after contrast'] = slots_after

    filename_out = os.path.splitext(filename)[0] + ' [contrast relations].csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


if __name__ == '__main__':
    align_slots(r'/d/Git/data2text-nlg/data/rest_e2e', 'devset.csv', E2EDataset, serialize_pos_info=False)

    # ----

    # score_slot_realizations(r'/d/Git/data2text-nlg/data/multiwoz',
    #                         'valid.csv', MultiWOZDataset, slot_level=True)
    # score_slot_realizations(r'/d/Git/data2text-nlg/data/video_game',
    #                         'valid.csv', ViggoDataset, slot_level=True)
    # score_slot_realizations(r'/d/Git/data2text-nlg/predictions_baselines/DataTuner/video_game',
    #                         'systemFc.csv', ViggoDataset, slot_level=True)

    # score_slot_realizations(r'/d/Git/data2text-nlg/predictions/video_game/finetuned_verbalized_slots/bart-base_lr_1e-5_bs_32_wus_100_run3/beam_search_20',
    #                         f'epoch_18_step_160_beam_search_reranked_att_1.0.csv', ViggoDataset, slot_level=True)
    # score_slot_realizations(r'/d/Git/data2text-nlg/predictions/rest_e2e/finetuned_verbalized_slots/bart-base_lr_1e-5_bs_32_wus_500_run3/beam_search_20',
    #                         f'epoch_17_step_2629_beam_search_reranked_att_1.0.csv', E2EDataset, slot_level=True)
    # score_slot_realizations(r'/d/Git/data2text-nlg/predictions/multiwoz/finetuned_verbalized_slots/bart-base_lr_1e-5_bs_32_wus_500_run4',
    #                         'epoch_18_step_1749_beam_search_reranked_att_2.0.csv', MultiWOZDataset, slot_level=True)

    # for thr in range(2, 5):
    #     score_slot_realizations(r'/d/Git/data2text-nlg/predictions/video_game/finetuned_verbalized_slots/bart-large_lr_4e-6_bs_16_wus_500_run1/att_guidance_eos_03',
    #                             f'epoch_13_step_319_beam_search_reranked_1.0_thr_{thr:02d}.csv', ViggoDataset, slot_level=True)
    # for thr in range(2, 5):
    #     score_slot_realizations(r'/d/Git/data2text-nlg/predictions/rest_e2e/finetuned_verbalized_slots/t5-base_lr_3e-5_bs_32_wus_100_run1/att_guidance_eos_03',
    #                             f'epoch_17_step_1315_beam_search_reranked_1.0_thr_{thr:02d}.csv', E2EDataset, slot_level=True)

    # ----

    # score_emphasis('predictions-rest_e2e_stylistic_selection/devset', 'predictions RNN (4+4) augm emph (reference).csv')

    # ----

    # predictions_dir = 'predictions rest_e2e (emphasis+contrast)'
    # predictions_file = 'predictions TRANS emphasis+contrast, train single, test combo extra (23.2k iter).csv'
    #
    # score_emphasis(predictions_dir, predictions_file)
    # score_contrast(predictions_dir, predictions_file)

    # ----

    # analyze_contrast_relations('rest_e2e', 'devset_e2e.csv')
