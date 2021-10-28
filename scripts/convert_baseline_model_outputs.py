import json
import os
import pandas as pd
import sys


def convert_datatuner_model_outputs(model_output_file):
    # Load the model's outputs
    with open(model_output_file, 'r', encoding='utf-8') as f_model_outputs:
        model_outputs = json.load(f_model_outputs)

    model_inputs = [output_dict['data'] for output_dict in model_outputs]
    model_texts = [output_dict['text'] for output_dict in model_outputs]

    # Combine the inputs and outputs in a dataframe, and remove duplicate rows
    df_converted = pd.DataFrame()
    df_converted['mr'] = model_inputs
    df_converted['utt'] = model_texts
    df_converted.drop_duplicates(inplace=True)

    # Compose the output file path
    out_dir = os.path.dirname(os.path.dirname(model_output_file))
    out_file_name = os.path.splitext(os.path.basename(model_output_file))[0] + '.csv'
    out_file_path = os.path.join(out_dir, out_file_name)

    # Save to a CSV file (with UTF-8-BOM encoding)
    df_converted.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    print(f'Saved to "{out_file_path}".')


def convert_tgen_model_outputs(model_output_file, dataset):
    # Load the corresponding dataset's test set
    if dataset == 'rest_e2e_cleaned':
        testset_file = os.path.join('data', 'rest_e2e_cleaned', 'test-fixed.csv')
    elif dataset == 'video_game':
        testset_file = os.path.join('data', 'video_game', 'test.csv')
    else:
        print(f'Error: dataset "{dataset}" not recognized')
        sys.exit()

    # Extract only the inputs and remove duplicates
    df_testset = pd.read_csv(testset_file, usecols=['mr'])
    df_testset.drop_duplicates(inplace=True)

    # Load the model's outputs
    with open(model_output_file, 'r', encoding='utf-8') as f_model_outputs:
        model_texts = f_model_outputs.read().splitlines()

    # Combine the inputs and outputs in a dataframe
    df_converted = pd.DataFrame()
    df_converted['mr'] = df_testset['mr'].tolist()
    df_converted['utt'] = model_texts

    # Compose the output file path
    out_dir = os.path.dirname(os.path.dirname(model_output_file))
    out_file_name = os.path.splitext(os.path.basename(model_output_file))[0] + '.csv'
    out_file_path = os.path.join(out_dir, out_file_name)

    # Save to a CSV file (with UTF-8-BOM encoding)
    df_converted.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    print(f'Saved to "{out_file_path}".')


if __name__ == '__main__':
    dataset = 'rest_e2e_cleaned'
    # dataset = 'video_game'
    model = 'DataTuner'
    # model = 'TGen+'

    if model == 'DataTuner':
        model_output_file = os.path.join('predictions_baselines', model, dataset, 'orig', 'systemFc.json')
        convert_datatuner_model_outputs(model_output_file)
    elif model == 'TGen+':
        model_output_file = os.path.join('predictions_baselines', model, dataset, 'orig', 'tgen-plus.run0.txt')
        convert_tgen_model_outputs(model_output_file, dataset)
    else:
        print('Error: model "{}" not recognized'.format(model))
        sys.exit()
