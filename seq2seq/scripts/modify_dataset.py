import os
import pandas as pd
import sys


def parse_da(mr, delimiters):
    # If no DA indication is expected in the data, return the MR unchanged
    if delimiters.get('da_beg') is None:
        return None

    # Verify if DA type is indicated at the beginning of the MR
    da_sep_idx = mr.find(delimiters['da_beg'])
    slot_sep_idx = mr.find(delimiters['slot_sep'])
    val_sep_idx = mr.find(delimiters['val_beg'])
    if da_sep_idx < 0 or 0 <= slot_sep_idx < da_sep_idx or 0 <= val_sep_idx < da_sep_idx:
        return None

    # Extract the DA type from the beginning of the MR
    return mr[:da_sep_idx]


def sample_rows(df_data, fraction):
    df_sampled = df_data.sample(frac=fraction)
    if len(df_sampled) < 1:
        df_sampled = df_data.sample(1)

    return df_sampled


def undersample_dataset(dataset, delimiters, fraction, trainset_only=False):
    if fraction <= 0.0 or fraction >= 1.0:
        print('Error: sample proportion must be greater than 0 and less than 1.')
        sys.exit()

    # Prepare the dataset file paths
    dataset_dir = os.path.join('..', 'data', dataset)
    if dataset == 'rest_e2e':
        data_files = ['trainset.csv']
        if not trainset_only:
            data_files.extend(['devset.csv', 'testset.csv'])
    elif dataset == 'video_game':
        data_files = ['train.csv']
        if not trainset_only:
            data_files.extend(['valid.csv', 'test.csv'])
    else:
        print(f'Error: dataset "{dataset}" not recognized')
        sys.exit()

    for file in data_files:
        # Load the data file
        df_data = pd.read_csv(os.path.join(dataset_dir, file), header=0)

        # Extract the DA types into a separate column
        df_data['da'] = df_data['mr'].apply(lambda x: parse_da(x, delimiters))

        # Sample the same fraction of rows from each DA type
        df_sampled = df_data.groupby('da', group_keys=False).apply(lambda x: sample_rows(x, fraction))
        df_sampled.drop('da', axis=1, inplace=True)

        # Compose the output file path
        out_file_name = os.path.splitext(file)[0] + f'_sampled_{fraction}.csv'
        out_file_path = os.path.join(dataset_dir, out_file_name)

        # Save to a CSV file (with UTF-8-BOM encoding)
        df_sampled.to_csv(out_file_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # dataset = 'rest_e2e'
    # delimiters = {
    #     'da_beg': None,
    #     'da_end': None,
    #     'slot_sep': ', ',
    #     'val_beg': '[',
    #     'val_end': ']'
    # }

    dataset = 'video_game'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    undersample_dataset(dataset, delimiters, 0.5, trainset_only=True)
