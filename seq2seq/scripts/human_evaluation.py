import os
import pandas as pd


def sample_from_data(data_file_path, sample_fraction=0.1):
    df_data = pd.read_csv(data_file_path, header=0, encoding='utf-8')
    df_sample = df_data.sample(frac=sample_fraction)

    print('>> Number of slots in the sample:')
    print(df_sample['mr'].str.count(r'\[').sum())

    out_file_path = os.path.splitext(data_file_path)[0] + f' [sample {sample_fraction * 100}%].csv'
    df_sample.to_csv(out_file_path, index=False, encoding='utf-8-sig')


def extract_utterances_with_errors(data_file_path):
    df_data = pd.read_csv(data_file_path, header=0, encoding='utf-8')
    df_errors = df_data[df_data['errors'] > 0]

    print('>> Number of errors:')
    print(df_errors['errors'].sum())

    out_file_path = os.path.splitext(data_file_path)[0] + f' [with errors only].csv'
    df_errors.to_csv(out_file_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # data_dir = 'seq2seq/predictions/video_game/finetuned_verbalized_slots/bart-base_lr_1e-5_bs_32_wus_100_run1'
    # data_file = 'epoch_10_step_160_no_beam_search.csv'
    # data_file = 'epoch_10_step_160_no_beam_search [errors].csv'

    # data_dir = 'seq2seq/predictions/rest_e2e/finetuned_verbalized_slots/bart-base_lr_1e-5_bs_32_wus_500_run1'
    # data_file = 'epoch_9_step_1315_beam_search_1.0.csv'
    # data_file = 'epoch_9_step_1315_beam_search_1.0 [errors].csv'

    data_dir = 'seq2seq/predictions/multiwoz/finetuned_verbalized_slots/t5-small_lr_2e-4_bs_64_wus_200_run3'
    # data_file = 'epoch_24_step_875_no_beam_search.csv'
    data_file = 'epoch_24_step_875_no_beam_search [errors].csv'

    # sample_from_data(os.path.join(data_dir, data_file), sample_fraction=0.04)
    extract_utterances_with_errors(os.path.join(data_dir, data_file))
