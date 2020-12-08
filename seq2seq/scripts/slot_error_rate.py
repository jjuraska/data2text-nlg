import os
import pandas as pd

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset


def calculate_slot_error_rate(data_dir, predictions_file, class_method, verbose=False):
    """Calculates the slot error rate in a file with generated utterances aligned with their corresponding input MRs.

    Slot realizations are only identified as exact slot value matches in utterances.
    """
    error_flags = []
    slot_sep = ' | '
    val_sep = ' = '

    df_data = pd.read_csv(os.path.join(data_dir, predictions_file), header=0)
    mrs_processed = class_method.preprocess_mrs(df_data.iloc[:, 0].to_list())
    predictions = df_data.iloc[:, 1].to_list()

    for mr, utt in zip(mrs_processed, predictions):
        values = []

        for slot_and_value_str in mr.split(slot_sep):
            # Ignore slots with no values
            if val_sep not in slot_and_value_str:
                continue

            # Parse the slot name and value
            slot_and_value = slot_and_value_str.split(val_sep)
            slot_name = slot_and_value[0].strip()
            value = slot_and_value[1].strip()

            # Ignore abstract slots and non-specific values
            if slot_name in ['topic', 'intent'] or value in ['?', 'none']:
                continue

            values.append(value)

        # Determine if the utterance has any errors (i.e., missing slot mentions)
        utt_lowercased = utt.lower()
        has_error = any(value.lower() not in utt_lowercased for value in values)
        error_flags.append(1 if has_error else 0)

    # Save the utterances along with their slot error indications to a CSV file
    df_data['has_err'] = error_flags
    out_file_path = os.path.splitext(os.path.join(data_dir, predictions_file))[0] + ' [errors].csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    ser = sum(error_flags) / len(predictions)

    # Print the SER
    if verbose:
        print(f'>> Slot error rate: {round(100 * ser, 2)}%')
    else:
        print(round(100 * ser, 2))

    return ser


def main():
    input_dir = os.path.join('..', 'predictions', 'multiwoz', 'finetuned_verbalized_slots', 'bart-base_lr_1e-5_bs_32_wus_500_run3')
    predictions_file = 'epoch_16_step_1749_beam_search_1.0.csv'
    class_method = MultiWOZDataset

    calculate_slot_error_rate(input_dir, predictions_file, class_method)


if __name__ == '__main__':
    main()
