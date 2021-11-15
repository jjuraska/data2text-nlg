import os

from data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset
from eval_utils import calculate_bleu
from scripts.slot_error_rate import calculate_slot_error_rate
from scripts.utterance_stats import utterance_stats
from slot_aligner.data_analysis import align_slots, score_slot_realizations


def batch_calculate_bleu(input_dir, dataset_class, verbose=False):
    files_processed = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt') and '_utt_only' in file_name:
            files_processed.append(file_name)
            if verbose:
                print(f'Running with file "{file_name}"...')

            calculate_bleu(os.path.join(input_dir, file_name), dataset_class.name, verbose=verbose)

            if verbose:
                print()

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def batch_calculate_slot_error_rate(input_dir, checkpoint_name, dataset_class, exact_matching=False, slot_level=False,
                                    verbose=False):
    files_processed = []
    ser_list = []

    decoding_suffixes = [
        # '_no_beam_search',
        # '_beam_search',
        '_nucleus_sampling',
        # '_beam_1.0_nucleus_sampling',
    ]
    was_reranking_used = True
    if 'gpt2' in os.path.split(input_dir)[-1]:
        length_penalty_vals = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    else:
        # length_penalty_vals = [0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
        length_penalty_vals = [1.0]
    p_vals = [0.3, 0.5, 0.8]

    for decoding_suffix in decoding_suffixes:
        reranking_suffixes = ['']
        if was_reranking_used and decoding_suffix != '_no_beam_search':
            reranking_suffixes.append('_reranked')
            reranking_suffixes.append('_reranked_att')

        for reranking_suffix in reranking_suffixes:
            if decoding_suffix == '_beam_search':
                value_suffixes = ['_' + str(val) for val in length_penalty_vals]
            elif decoding_suffix in ['_nucleus_sampling', '_beam_1.0_nucleus_sampling']:
                value_suffixes = ['_' + str(val) for val in p_vals]
            else:
                value_suffixes = ['']

            for value_suffix in value_suffixes:
                file_name = checkpoint_name + decoding_suffix + reranking_suffix + value_suffix + '.csv'
                files_processed.append(file_name)
                if verbose:
                    print(f'Running with file "{file_name}"...')

                if exact_matching:
                    ser = calculate_slot_error_rate(input_dir, file_name, dataset_class, slot_level=slot_level,
                                                    verbose=verbose)
                else:
                    ser = score_slot_realizations(input_dir, file_name, dataset_class, slot_level=slot_level,
                                                  verbose=verbose)
                ser_list.append(ser)

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def batch_find_slot_alignment(input_dir, file_names, dataset_class, serialize_pos_info=False, verbose=False):
    files_processed = []

    for file_name in file_names:
        files_processed.append(file_name)
        if verbose:
            print(f'Running with file "{file_name}"...')

        align_slots(input_dir, file_name, dataset_class, serialize_pos_info=serialize_pos_info)

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def batch_utterance_stats(input_dir, export_vocab=False, verbose=False):
    files_processed = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and '[errors' not in file_name:
            files_processed.append(file_name)
            if verbose:
                print(f'Running with file "{file_name}"...')

            utterance_stats(os.path.join(input_dir, file_name), export_vocab=export_vocab, verbose=verbose)

            if verbose:
                print()

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def run_batch_calculate_bleu():
    input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots', 'bart-base_lr_1e-5_bs_32_wus_500_run4')
    dataset_class = MultiWOZDataset

    batch_calculate_bleu(input_dir, dataset_class, verbose=False)


def run_batch_calculate_slot_error_rate():
    input_dir = os.path.join('predictions', 'video_game', 'finetuned_verbalized_slots',
                             't5-base_lr_3e-5_bs_16_wus_100_run3')
    checkpoint_name = 'epoch_16_step_319'
    dataset_class = ViggoDataset

    # input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots',
    #                          'bart-base_lr_1e-5_bs_32_wus_500_run4')
    # checkpoint_name = 'epoch_18_step_1749'
    # dataset_class = MultiWOZDataset

    # batch_calculate_slot_error_rate(
    #     input_dir, checkpoint_name, dataset_class, exact_matching=True, slot_level=False, verbose=False)
    batch_calculate_slot_error_rate(
        input_dir, checkpoint_name, dataset_class, exact_matching=False, slot_level=True, verbose=False)


def run_batch_find_slot_alignment():
    # input_dir = os.path.join('data', 'rest_e2e')
    # file_names = ['trainset.csv', 'devset.csv', 'testset.csv']
    # dataset_class = E2EDataset

    # input_dir = os.path.join('data', 'rest_e2e_cleaned')
    # file_names = ['train-fixed.no-ol.csv', 'devel-fixed.no-ol.csv', 'test-fixed.csv']
    # dataset_class = E2ECleanedDataset

    input_dir = os.path.join('data', 'multiwoz')
    file_names = ['train.csv', 'valid.csv', 'test.csv']
    dataset_class = MultiWOZDataset

    # input_dir = os.path.join('data', 'video_game')
    # file_names = ['train.csv', 'valid.csv', 'test.csv']
    # dataset_class = ViggoDataset

    batch_find_slot_alignment(input_dir, file_names, dataset_class, serialize_pos_info=False, verbose=True)


def run_batch_utterance_stats():
    # input_dir = os.path.join('predictions_baselines', 'DataTuner', 'video_game')
    # input_dir = os.path.join('predictions', 'rest_e2e_cleaned', 'finetuned', 'gpt2_lr_2e-5_bs_20_wus_500_run1')
    input_dir = os.path.join('predictions', 'video_game', 'finetuned', 'gpt2_lr_2e-5_bs_16_wus_100_run4')
    # input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots', 't5-small_lr_2e-4_bs_64_wus_200_run3')

    batch_utterance_stats(input_dir, export_vocab=False, verbose=False)


if __name__ == '__main__':
    # run_batch_calculate_bleu()
    run_batch_calculate_slot_error_rate()
    # run_batch_find_slot_alignment()
    # run_batch_utterance_stats()
