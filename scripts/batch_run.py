import os

from dataset_loaders.e2e import E2EDataset, E2ECleanedDataset
from dataset_loaders.multiwoz import MultiWOZDataset
from dataset_loaders.viggo import ViggoDataset
from eval_utils import calculate_bleu, calculate_bertscore
from scripts.slot_error_rate import calculate_slot_error_rate
from scripts.utterance_stats import get_utterance_stats
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


def batch_calculate_slot_error_rate(input_dir, dataset_class, exact_matching=False, slot_level=False, verbose=False):
    files_processed = []
    ser_list = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and '[' not in file_name:
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


def batch_utterance_stats(input_dir, dataset_class, export_delex=False, export_vocab=False, verbose=False):
    files_processed = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and '[' not in file_name:
            file_path = os.path.join(input_dir, file_name)
            files_processed.append(file_name)
            if verbose:
                print(f'Running with file "{file_name}"...')

            stats = get_utterance_stats(file_path, export_vocab=export_vocab, verbose=verbose)
            stats_delex = get_utterance_stats(file_path, mr_file_path=file_path, dataset_class=dataset_class,
                                              export_delex=export_delex, export_vocab=export_vocab, verbose=verbose)

            # Print statement to interweave the results on lex and delex utterances
            # stats_pairs = [f'{score}\t{score_delex}'
            #                for score, score_delex in zip(stats.values(), stats_delex.values())]
            # print('\t'.join([str(value) for value in stats_pairs]))

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
                             't5-small_lr_2e-4_bs_32_wus_100_run3', 'diverse_beam_search_10')
    dataset_class = ViggoDataset

    # input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots',
    #                          'bart-base_lr_1e-5_bs_32_wus_500_run4')
    # dataset_class = MultiWOZDataset

    # batch_calculate_slot_error_rate(input_dir, dataset_class, exact_matching=True, slot_level=False, verbose=False)
    batch_calculate_slot_error_rate(input_dir, dataset_class, exact_matching=False, slot_level=True, verbose=False)


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
    # input_dir = os.path.join('predictions_baselines', 'Slug2Slug', 'rest_e2e')
    # dataset_class = E2EDataset

    # input_dir = os.path.join('predictions', 'rest_e2e', 'finetuned_verbalized_slots',
    #                          't5-small_lr_2e-4_bs_64_wus_100_run1')
    # dataset_class = E2EDataset

    # input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots',
    #                          'bart-base_lr_1e-5_bs_32_wus_500_run4')
    # dataset_class = MultiWOZDataset

    input_dir = os.path.join('predictions', 'video_game', 'finetuned_verbalized_slots',
                             't5-small_lr_2e-4_bs_32_wus_100_run3', 'diverse_beam_search_10')
    dataset_class = ViggoDataset

    batch_utterance_stats(input_dir, dataset_class=dataset_class, export_delex=False, export_vocab=False, verbose=False)


if __name__ == '__main__':
    # run_batch_calculate_bleu()
    # run_batch_calculate_slot_error_rate()
    # run_batch_find_slot_alignment()
    run_batch_utterance_stats()
