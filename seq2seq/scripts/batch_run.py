import os

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset
from seq2seq.scripts.slot_error_rate import calculate_slot_error_rate
from seq2seq.scripts.utterance_stats import utterance_stats


def batch_calculate_slot_error_rate(input_dir, checkpoint_name, class_method, verbose=False):
    files_processed = []
    ser_list = []

    decoding_suffixes = [
        '_no_beam_search',
        '_beam_search',
        # '_nucleus_sampling',
    ]
    was_reranking_used = False
    if 'gpt2' in os.path.split(input_dir)[-1]:
        length_penalty_vals = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    else:
        length_penalty_vals = [0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
    p_vals = [0.3, 0.5, 0.8]

    for decoding_suffix in decoding_suffixes:
        reranking_suffixes = ['']
        if was_reranking_used and decoding_suffix != '_no_beam_search':
            reranking_suffixes.append('_reranked')

        for reranking_suffix in reranking_suffixes:
            if decoding_suffix == '_beam_search':
                value_suffixes = ['_' + str(val) for val in length_penalty_vals]
            elif decoding_suffix == '_nucleus_sampling':
                value_suffixes = ['_' + str(val) for val in p_vals]
            else:
                value_suffixes = ['']

            for value_suffix in value_suffixes:
                file_name = checkpoint_name + decoding_suffix + reranking_suffix + value_suffix + '.csv'
                files_processed.append(file_name)
                if verbose:
                    print(f'Running with file "{file_name}"...')

                ser = calculate_slot_error_rate(input_dir, file_name, class_method, verbose=verbose)
                ser_list.append(ser)

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def batch_utterance_stats(input_dir, export_vocab=False, verbose=False):
    files_processed = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and '[errors]' not in file_name:
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


def run_batch_score_slot_realizations(verbose=False):
    input_dir = os.path.join('..', 'predictions', 'multiwoz', 'finetuned_verbalized_slots', 't5-small_lr_2e-4_bs_64_wus_200_run3')
    checkpoint_name = 'epoch_24_step_875'
    class_method = MultiWOZDataset

    batch_calculate_slot_error_rate(input_dir, checkpoint_name, class_method, verbose=verbose)


def run_batch_utterance_stats(verbose=False):
    # input_dir = os.path.join('..', 'predictions_baselines', 'DataTuner', 'video_game')
    # input_dir = os.path.join('..', 'predictions', 'rest_e2e_cleaned', 'finetuned', 'gpt2_lr_2e-5_bs_20_wus_500_run1')
    input_dir = os.path.join('..', 'predictions', 'video_game', 'finetuned', 'gpt2_lr_2e-5_bs_16_wus_100_run4')

    batch_utterance_stats(input_dir, export_vocab=False, verbose=verbose)


if __name__ == '__main__':
    run_batch_score_slot_realizations(verbose=False)
    # run_batch_utterance_stats(verbose=False)
