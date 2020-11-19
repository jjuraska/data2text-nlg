from itertools import chain
import os
import pandas as pd
import re
# from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from tqdm import tqdm

from seq2seq.slot_aligner.slot_alignment import score_alignment


def calculate_singleref_bleu(dataset, predictions):
    """Calculates the corpus BLEU score with a single reference per generated utterance.

    If the dataset is grouped by MR, and thus has a list of reference utterances for each MR, this method flattens the
    references and multiplies the generated predictions as necessary to match corresponding references. Otherwise, it
    uses the predictions and references as-is.
    """
    references = dataset.get_utterances(lowercased=True)

    if isinstance(references[0], list):
        # Multiply generated utterances depending on the number of corresponding references, and then flatten references
        predictions_extended = list(chain.from_iterable(
            [pred] * len(ref_list) for pred, ref_list in zip(predictions, references)))
        references_flat = list(chain.from_iterable(references))
    else:
        predictions_extended = predictions
        references_flat = references

    return corpus_bleu(predictions_extended, [references_flat]).score


def calculate_multiref_bleu(dataset, predictions):
    """Calculates the corpus BLEU score with multiple references per generated utterance.

    Assumes the dataset to be grouped by MR, and to thus have a list of reference utterances for each MR. Assumes the
    generated utterances to have been produced from unique inputs, and hence to be a flat list. This method transposes
    the nested list of reference utterances to conform with the format sacreblue's corpus_bleu method expects.
    """
    references = dataset.get_utterances(lowercased=True)

    # Only works if the number of references is the same for each input
    # references_transposed = list(map(list, zip(*references)))

    # Transpose the reference utterances
    max_num_refs = max(len(ref_list) for ref_list in references)
    references_transposed = [[] for _ in range(max_num_refs)]
    for ref_list in references:
        idx = 0
        for ref in ref_list:
            references_transposed[idx].append(ref)
            idx += 1

        # Pad with the first reference
        for i in range(idx, max_num_refs):
            references_transposed[i].append(ref_list[0])

    return corpus_bleu(predictions, references_transposed).score


def rerank_beams(beams, mrs, keep_n=None, keep_least_errors_only=False):
    """Reranks beams based on the slot error rate determined by the slot aligner. Keeps at most n best candidates.

    Note: Python's sort is guaranteed to be stable, i.e., when multiple records have the same key (e.g., slot error
    score), their original order (e.g., based on their beam score) is preserved.
    """
    beams_reranked = []

    for idx, mr in enumerate(tqdm(mrs, desc='Reranking')):
        beam_scored = []

        for utt in beams[idx]:
            # Calculate the slot error score
            score = score_alignment(utt, mr)
            beam_scored.append((utt, score))

        # Rerank utterances by slot error score (the higher the better)
        beam_scored.sort(key=lambda tup: tup[1], reverse=True)

        if keep_least_errors_only:
            # Filter only those utterances that have the least number of errors identified by the slot aligner
            beam_scored = [candidate for candidate in beam_scored if candidate[1] == beam_scored[0][1]]

        # Keep at most n candidates
        if keep_n is not None and len(beam_scored) > keep_n > 0:
            beam_scored = beam_scored[:keep_n]

        # DEBUG
        # if idx < 5:
        #     print('>> Scored beams:')
        #     print('\n'.join('{0} :: {1}'.format(utt[1], utt[0]) for utt in beam_scored))
        #     print()

        # Store the reranked beam (utterances only)
        beams_reranked.append([utt[0] for utt in beam_scored])

    return beams_reranked


def execute_e2e_evaluation_script(config, test_set, eval_configurations):
    """Runs the evaluation script of the E2E NLG Challenge for multiple sets of generated utterances.

    Metrics the utterances are evaluated on: BLEU, NIST, METEOR, ROUGE-L, CIDEr.
    """
    scores = {}

    # Make sure the output directory exists for the given dataset
    predictions_dir = os.path.join('seq2seq', 'predictions', test_set.name)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Prepare the metrics script command, and create the reference file for the given dataset
    eval_dir = os.path.join('seq2seq', 'eval')
    metrics_script = 'python ' + os.path.join(eval_dir, 'E2E', 'measure_scores.py')
    reference_file = os.path.join(eval_dir, 'test_references_{}.txt'.format(test_set.name))
    if not os.path.exists(reference_file):
        print('>> Generating a reference file for the "{}" test set.'.format(test_set.name))
        test_set.create_reference_file_for_testing()

    for prediction_list, reranked in eval_configurations:
        file_name_root = compose_output_file_name(config, reranked=reranked)

        # Save generated utterances along with their corresponding MRs into a CSV file
        file_name = f'{file_name_root}.csv'
        df_predictions = pd.DataFrame({'mr': test_set.get_mrs(raw=True), 'utt': prediction_list})
        df_predictions.to_csv(os.path.join(predictions_dir, file_name), index=False, encoding='utf-8-sig')

        # Save generated utterances in a text file (for reference-based metric evaluation)
        file_name = f'{file_name_root}_utt_only.txt'
        predictions_file = os.path.join(predictions_dir, file_name)
        with open(predictions_file, 'w') as f_out:
            for prediction in prediction_list:
                f_out.write(prediction + '\n')

        # Run the metrics script provided by the E2E NLG Challenge
        print('Running E2E evaluation script...')
        script_output = os.popen(metrics_script + ' ' + reference_file + ' ' + predictions_file).read()
        scores_key = 'reranked' if reranked else 'not_reranked'
        scores[scores_key] = parse_scores_from_e2e_script_output(script_output)

        # Print the scores
        print()
        print('\n'.join([f'{metric}: {score}' for metric, score in scores[scores_key]]))
        print()

    return scores


def parse_scores_from_e2e_script_output(script_output):
    """Extracts the individual metric scores from the E2E NLG Challenge evaluation script's output.

    Returns:
        A list of (metric_name: str, score: float) tuples.
    """
    match = re.search(r'BLEU: (?P<BLEU>\d+\.\d+)\s+NIST: (?P<NIST>\d+\.\d+)\s+METEOR: (?P<METEOR>\d+\.\d+)\s+'
                      r'ROUGE_L: (?P<ROUGE_L>\d+\.\d+)\s+CIDEr: (?P<CIDEr>\d+\.\d+)', script_output)

    scores = [(metric, float(match.group(metric))) for metric in ['BLEU', 'METEOR', 'ROUGE_L', 'CIDEr']]

    return scores


def update_test_scores(scores_dict, new_scores_dict):
    """Appends scores of a test run with a new test configuration to the scores from all previous runs."""
    for key in scores_dict:
        if key in new_scores_dict:
            scores_dict[key].append(new_scores_dict[key])


def print_test_scores(scores_dict, output_dir=None):
    """Prints the final summary of scores from all test runs as a table."""
    scores_str = ''

    print()
    print(' ************************ ')
    print('**  TEST SCORE SUMMARY  **')
    print(' ************************ ')
    print()

    for key in ['not_reranked', 'reranked']:
        if scores_dict.get(key):
            scores_str += f'---- {key} ----\n'
            scores_str += '\t'.join([metric for metric, val in scores_dict[key][0]]) + '\n'
            scores_str += '\n'.join(['\t'.join([f'{val:.4f}' for metric, val in score_list])
                                     for score_list in scores_dict[key]]) + '\n\n'

    print(scores_str, end='')

    if output_dir is not None:
        with open(os.path.join(output_dir, 'test_scores.txt'), 'a') as f_out:
            f_out.write(scores_str)


def print_best_checkpoints(checkpoints):
    print()
    print(' ********************** ')
    print('**  BEST CHECKPOINTS  **')
    print(' ********************** ')
    print()

    best_checkpoint_str = ''
    for metric in ['loss', 'perplexity', 'BLEU', 'BLEU (multi-ref)']:
        best_checkpoint_str += '>> Validation {}: {:.4f} (epoch {}, step {})'.format(
            metric, checkpoints[metric][2], checkpoints[metric][0], checkpoints[metric][1]) + '\n'

    print(best_checkpoint_str, end='')

    model_dir = os.path.join('seq2seq', 'model')
    with open(os.path.join(model_dir, 'best_checkpoints.txt'), 'a') as f_out:
        f_out.write(best_checkpoint_str)


def compose_output_file_name(config, reranked=False):
    if config.num_beams > 1:
        inference_method_suffix = '_beam_search_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.length_penalty)
    elif config.do_sample and config.top_p < 1.0:
        inference_method_suffix = '_nucleus_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_p)
    elif config.do_sample and config.top_k > 0:
        inference_method_suffix = '_top_k_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_k)
    else:
        inference_method_suffix = '_no_beam_search'

    file_name = 'epoch_{}_step_{}{}'.format(config.checkpoint_epoch, config.checkpoint_step, inference_method_suffix)

    return file_name
