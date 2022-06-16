import os
import pandas as pd

from dataset_loaders.e2e import E2EDataset, E2ECleanedDataset
from dataset_loaders.multiwoz import MultiWOZDataset
from dataset_loaders.viggo import ViggoDataset


def calculate_slot_error_rate(data_dir, predictions_file, dataset_class, slot_level=False, verbose=False):
    """Calculates the slot error rate in a file with generated utterances aligned with their corresponding input MRs.

    Slot realizations are only identified as exact slot value matches in utterances.
    """
    error_flags = []

    # Load the input MRs and output utterances
    df_data = pd.read_csv(os.path.join(data_dir, predictions_file), header=0)
    mrs_raw = df_data.iloc[:, 0].to_list()
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False)
    predictions = df_data.iloc[:, 1].to_list()

    for mr_as_list, utt in zip(mrs_processed, predictions):
        # Ignore abstract slots and non-specific values
        mr_as_list = [(slot, value) for slot, value in mr_as_list
                      if slot != 'da' and value not in ['', '?', 'none']]

        # Determine if the utterance has any errors (i.e., missing slot mentions)
        utt_lowercased = utt.lower()
        has_error = any(value.lower() not in utt_lowercased for value in map(lambda x: x[1], mr_as_list))
        error_flags.append(1 if has_error else 0)

    # Save the utterances along with their slot error indications to a CSV file
    df_data['has_err'] = error_flags
    out_file_path = os.path.splitext(os.path.join(data_dir, predictions_file))[0] + ' [errors (exact-match)].csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    # Calculate the slot-level or utterance-level SER
    if slot_level:
        raise NotImplementedError('Slot-level SER evaluation not supported yet')
    else:
        ser = sum(error_flags) / len(predictions)

    # Print the SER
    if verbose:
        print(f'>> Slot error rate: {round(100 * ser, 2)}%')
    else:
        print(f'{round(100 * ser, 2)}%')

    return ser


def main():
    # input_dir = os.path.join('predictions', 'multiwoz', 'finetuned_verbalized_slots',
    #                          'bart-base_lr_1e-5_bs_32_wus_500_run3')
    # predictions_file = 'epoch_16_step_1749_beam_search_1.0.csv'
    input_dir = os.path.join('data', 'multiwoz')
    predictions_file = 'test.csv'
    dataset_class = MultiWOZDataset

    calculate_slot_error_rate(input_dir, predictions_file, dataset_class, slot_level=False, verbose=False)


if __name__ == '__main__':
    main()
