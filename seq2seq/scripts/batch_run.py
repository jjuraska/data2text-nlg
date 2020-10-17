import os

from seq2seq.scripts.utterance_stats import utterance_stats


# Absolute path from the project root
input_dir = os.path.join('seq2seq', 'predictions', 'video_game', 't5-small_lr_1e-4_bs_12_wus_400')

for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        print(f'Running with file "{file_name}"...')
        utterance_stats(os.path.join(input_dir, file_name), export_vocab=True)
        print()
