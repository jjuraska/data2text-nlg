import os

from seq2seq.scripts.utterance_stats import utterance_stats


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


if __name__ == '__main__':
    # input_dir = os.path.join('..', 'predictions_baselines', 'DataTuner', 'video_game')
    # input_dir = os.path.join('..', 'predictions', 'rest_e2e_cleaned', 'finetuned', 'gpt2_lr_2e-5_bs_20_wus_500_run1')
    input_dir = os.path.join('..', 'predictions', 'video_game', 'finetuned', 'gpt2_lr_2e-5_bs_16_wus_100_run4')

    batch_utterance_stats(input_dir, export_vocab=False, verbose=False)
