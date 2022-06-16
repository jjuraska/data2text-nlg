from itertools import chain
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re
# from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from tqdm import tqdm

from eval.RNNLG.GentScorer import GentScorer
from slot_aligner.slot_alignment import count_errors


def calculate_singleref_bleu(dataset, predictions):
    """Calculates the corpus BLEU score with a single reference per generated utterance.

    If the dataset is grouped by MR, and thus has a list of reference utterances for each MR, this method flattens the
    references and multiplies the generated predictions as necessary to match corresponding references. Otherwise, it
    uses the predictions and references as-is.
    """
    references = dataset.get_utterances(lowercase=True)
    predictions = [pred.lower() for pred in predictions]

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
    references = dataset.get_utterances(lowercase=True)
    predictions = [pred.lower() for pred in predictions]

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


def calculate_bleu(predictions_file, dataset_name, verbose=False):
    """Calculates the BLEU score using sacreblue, as well as the RNNLG script.

    Note: Currently only works with datasets that do not have multiple references.
    """
    # Initialize the RNNLG scorer
    gentscorer = GentScorer(os.path.join('eval', 'RNNLG', 'detect.pair'))

    # Load generated utterances and the corresponding reference utterances
    with open(predictions_file, 'r', encoding='utf-8') as f_pred:
        utterances = [pred_line.strip() for pred_line in f_pred.readlines()]

    references_file = os.path.join('eval', f'test_references_{dataset_name}.txt')
    with open(references_file, 'r', encoding='utf-8') as f_ref:
        references = [ref_line.strip() for ref_line in f_ref.readlines() if ref_line.strip() != '']

    # Preprocess generated utterances, as well as references
    utterances_lower = [utt.lower() for utt in utterances]
    references_lower = [ref.lower() for ref in references]
    parallel_corpus_tokenized = [[[' '.join(word_tokenize(utt))], [' '.join(word_tokenize(ref))]]
                                 for utt, ref in zip(utterances_lower, references_lower)]

    bleu_sacre = corpus_bleu(utterances_lower, [references_lower]).score
    bleu_rnnlg = gentscorer.scoreSBLEU(parallel_corpus_tokenized)

    # Print the BLEU scores
    if verbose:
        print(f'>> BLEU (sacreblue): {round(bleu_sacre / 100, 4)}')
        print(f'>> BLEU (RNNLG):     {round(bleu_rnnlg, 4)}')
    else:
        print(f'{round(bleu_sacre / 100, 4)}\t{round(bleu_rnnlg, 4)}')


def rerank_beams(beams, mrs, domain, keep_n=None, keep_least_errors_only=False):
    """Reranks beams based on the slot error rate determined by the slot aligner. Keeps at most n best candidates.

    Note: Python's sort is guaranteed to be stable, i.e., when multiple records have the same key (e.g., slot error
    score), their original order (e.g., based on their beam score) is preserved.
    """
    beams_reranked = []

    for idx, mr in enumerate(tqdm(mrs, desc='Reranking')):
        beam_scored = []

        for utt in beams[idx]:
            # Calculate the slot error score
            num_errors, _, _, _ = count_errors(utt, mr, domain)
            score = 1 / (num_errors + 1)
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


def rerank_beams_attention_based(beams, slot_errors):
    beams_reranked = []
    slot_errors_reranked = []

    for beam_utts, beam_errors in tqdm(zip(beams, slot_errors), desc='Reranking'):
        beam_scored = list(zip(beam_utts, beam_errors))

        # Rerank utterances by the number of slot errors (the lower the better)
        beam_scored.sort(key=lambda tup: len(tup[1]))

        # DEBUG
        # print(beam_scored)
        # print()

        # Store the reranked beam utterances as well as slot errors
        beams_reranked.append([utt[0] for utt in beam_scored])
        slot_errors_reranked.append([utt[1] for utt in beam_scored])

    return beams_reranked, slot_errors_reranked


def execute_e2e_evaluation_script(config, test_set, eval_configurations):
    """Runs the evaluation script of the E2E NLG Challenge for multiple sets of generated utterances.

    Metrics the utterances are evaluated on: BLEU, NIST, METEOR, ROUGE-L, CIDEr.
    """
    scores = {}

    # Make sure the output directory exists for the given dataset
    predictions_dir = os.path.join('predictions', test_set.name)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Prepare the metrics script command, and create the reference file for the given dataset
    metrics_script = 'python ' + os.path.join('eval', 'E2E', 'measure_scores.py')
    reference_file = os.path.join('eval', 'test_references_{}.txt'.format(test_set.name))
    if not os.path.exists(reference_file):
        print('>> Generating a reference file for the "{}" test set.'.format(test_set.name))
        test_set.create_reference_file_for_testing()

    for prediction_list, reranked, slot_errors in eval_configurations:
        file_name_root = compose_output_file_name(config, reranked=reranked, attention_based=slot_errors is not None)

        # Save generated utterances along with their corresponding MRs to a CSV file
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
        scores_key = ('reranked_att' if slot_errors is not None else 'reranked') if reranked else 'not_reranked'
        scores[scores_key] = parse_scores_from_e2e_script_output(script_output)

        # Print the scores
        print()
        print('\n'.join([f'{metric}: {score}' for metric, score in scores[scores_key]]))
        print()

    return scores


def save_slot_errors(config, test_set, eval_configurations):
    """Saves generated utterances along with the attention-based slot errors, as well as the input MRs, to a file."""

    # Make sure the output directory exists for the given dataset
    predictions_dir = os.path.join('predictions', test_set.name)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    for prediction_list, reranked, slot_errors in eval_configurations:
        # Skip configurations without slot error indications (such as when using the slot aligner for reranking)
        if slot_errors is None:
            continue

        file_name_root = compose_output_file_name(config, reranked=reranked, attention_based=True)

        # Save generated utterances along with their corresponding MRs and slot errors to a CSV file
        file_name = f'{file_name_root} [errors (attention-based)].csv'
        df_predictions = pd.DataFrame({
            'mr': test_set.get_mrs(raw=True),
            'utt': prediction_list,
            'errors': [len(slot_error_list) for slot_error_list in slot_errors],
            'incorrect slots': [', '.join(slot_error_list) for slot_error_list in slot_errors]
        })
        df_predictions.to_csv(os.path.join(predictions_dir, file_name), index=False, encoding='utf-8-sig')


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

    for key in ['not_reranked', 'reranked', 'reranked_att']:
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

    with open(os.path.join('model', 'best_checkpoints.txt'), 'a') as f_out:
        f_out.write(best_checkpoint_str)


def compose_output_file_name(config, reranked=False, attention_based=False):
    decoding_method_setting = None

    if config.num_beams > 1:
        if config.do_sample:
            decoding_method = 'beam_' + str(config.length_penalty)
            if config.num_beam_groups > 1:
                decoding_method = 'diverse_' + str(config.diversity_penalty) + '_' + decoding_method
            if config.top_p < 1.0:
                decoding_method += '_nucleus_sampling'
                decoding_method_setting = str(config.top_p)
            elif config.top_k > 0:
                decoding_method += '_top_k_sampling'
                decoding_method_setting = str(config.top_k)
        else:
            decoding_method = 'beam_search'
            if config.num_beam_groups > 1:
                decoding_method = 'diverse_' + str(config.diversity_penalty) + '_' + decoding_method
            decoding_method_setting = str(config.length_penalty)
    elif config.do_sample and config.top_p < 1.0:
        decoding_method = 'nucleus_sampling'
        decoding_method_setting = str(config.top_p)
    elif config.do_sample and config.top_k > 0:
        decoding_method = 'top_k_sampling'
        decoding_method_setting = str(config.top_k)
    else:
        decoding_method = 'greedy_search'

    # Compose the suffix based on the decoding method
    suffix = decoding_method
    if reranked:
        suffix += '_reranked'
        if attention_based:
            suffix += '_att'
    if decoding_method_setting:
        suffix += '_' + decoding_method_setting

    file_name = 'epoch_{}_step_{}_{}'.format(config.checkpoint_epoch, config.checkpoint_step, suffix)

    return file_name
