import bert_score
from bleurt.score import LengthBatchingBleurtScorer
from datasets import load_metric
from datasets.metric import Metric
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import random
from sacrebleu import corpus_bleu
import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Type, Union

from data_loader import MRToTextDataset
from dataset_loaders.e2e import E2EDataset, E2ECleanedDataset
from dataset_loaders.multiwoz import MultiWOZDataset
from dataset_loaders.viggo import ViggoDataset
from scripts.perturbate_utterances import PerturbationMode, PseudoUtteranceGenerator, SlotNameMode


class BleurtModelPath(object):
    BLEURT_20 = 'eval/BLEURT/models/BLEURT-20'
    BLEURT_20_D12 = 'eval/BLEURT/models/BLEURT-20-D12'
    BLEURT_20_D6 = 'eval/BLEURT/models/BLEURT-20-D6'
    BLEURT_20_D3 = 'eval/BLEURT/models/BLEURT-20-D3'


class BertScoreModelCheckpoint(object):
    DEBERTA_XLARGE_MNLI = 'microsoft/deberta-xlarge-mnli'
    DEBERTA_LARGE_MNLI = 'microsoft/deberta-large-mnli'


class PseudoReferenceMetricEvaluator(object):
    def __init__(self, dataset_name: str, partition: Optional[str] = None, file_path: Optional[str] = None,
                 separator: str = ' ', include_slot_names: bool = False,
                 bool_slot_name_mode: str = SlotNameMode.VERBALIZED, lowercase: bool = False,
                 perturbation: Optional[str] = None, perturbation_n: int = 1) -> None:
        self.dataset_class = self.get_dataset_class(dataset_name)
        self.partition = partition
        self.file_path = file_path
        self.separator = separator
        self.include_slot_names = include_slot_names
        self.bool_slot_name_mode = bool_slot_name_mode
        self.lowercase = lowercase
        self.perturbation = perturbation
        self.perturbation_n = perturbation_n

        self.generator = PseudoUtteranceGenerator(self.dataset_class, partition=partition, file_path=file_path)
        self.predictions = [utt.lower() for utt in self.generator.utterances] if lowercase else self.generator.utterances
        self.references = None
        self.generate_new_references()

    @staticmethod
    def get_dataset_class(dataset_name: str) -> Type[MRToTextDataset]:
        if dataset_name == 'rest_e2e':
            return E2EDataset
        elif dataset_name == 'rest_e2e_cleaned':
            return E2ECleanedDataset
        elif dataset_name == 'multiwoz':
            return MultiWOZDataset
        elif dataset_name == 'video_game':
            return ViggoDataset
        else:
            raise ValueError(f'Dataset "{dataset_name}" not recognized')

    def generate_new_references(self) -> None:
        self.references = self.generator.generate_pseudo_utterances(
            separator=self.separator, include_slot_names=self.include_slot_names,
            bool_slot_name_mode=self.bool_slot_name_mode, lowercase=self.lowercase, perturbation=self.perturbation,
            perturbation_n=self.perturbation_n)

        # NOTE: For experiments only
        # self.references = self.generator.generate_mrs(
        #     exclude_da=False, lowercase=self.lowercase, perturbation=self.perturbation,
        #     perturbation_n=self.perturbation_n)

    def shuffle_references(self, within_da_only: bool = False) -> None:
        if within_da_only:
            refs_shuffled = []
            das = []
            for mr in self.generator.mrs:
                for slot_name, slot_value in mr:
                    if slot_name == 'da':
                        das.append(slot_value)
                        break

            if len(das) != len(self.references):
                raise ValueError('Cannot shuffle within DA only. Some meaning representations appear not to have the DA indicated.')

            # Group references by DA in a dictionary
            refs_by_da = defaultdict(list)
            for ref, da in zip(self.references, das):
                refs_by_da[da].append(ref)
            # Shuffle references within each DA
            for da in refs_by_da:
                random.shuffle(refs_by_da[da])
            # Create a list of references that follow the original DA order
            for da in das:
                refs_shuffled.append(refs_by_da[da].pop())

            self.references = refs_shuffled
        else:
            random.shuffle(self.references)

    def get_csv_export_path(self, suffix: Optional[str] = None) -> str:
        in_file_path = self.dataset_class.get_data_file_path(self.partition) if self.partition else self.file_path

        dir_path = os.path.join(os.path.dirname(in_file_path), 'pseudo_references')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = f'{self.partition}_' if self.partition else ''
        if self.perturbation:
            file_name += f'{self.perturbation}{self.perturbation_n}'
        else:
            file_name += 'no_pert'

        if self.separator == ' ':
            file_name += '_space_sep'
        elif self.separator == ', ':
            file_name += '_comma_sep'
        elif self.separator == '. ':
            file_name += '_period_sep'
        else:
            file_name += '_other_sep'

        if self.lowercase:
            file_name += '_lowercase'

        if suffix:
            file_name += suffix

        file_name += '.csv'

        return os.path.join(dir_path, file_name)

    def calculate_all_metrics(self, num_runs: int = 1, shuffle_refs: Union[bool, str] = False,
                              export_results: bool = False, **kwargs: Any) -> None:
        results = []

        # Initialize scorers for metrics that depend on a model or a package, so they don't get loaded in every run
        meteor_scorer = self.init_meteor_scorer()
        bert_scorer = self.init_bert_scorer(
            model=kwargs.get('bertscore_model', None),
            batch_size=kwargs.get('bertscore_batch_size', 64),
            idf=kwargs.get('bertscore_idf', False))
        bleurt_scorer = self.init_bleurt_scorer(model=kwargs.get('bleurt_model', None))

        for run in tqdm(range(1, num_runs + 1), desc='Run'):
            # Shuffle references, if desired
            if shuffle_refs:
                self.shuffle_references(within_da_only=(shuffle_refs == 'da'))

            # Calculate segment-level scores
            bleu_results = self.calculate_bleu()
            meteor_results = self.calculate_meteor(meteor_scorer)
            rouge_results = self.calculate_rouge(use_stemmer=kwargs.get('rouge_use_stemmer', False))
            bertscore_results = self.calculate_bertscore(bert_scorer)
            bleurt_results = self.calculate_bleurt(bleurt_scorer, batch_size=kwargs.get('bleurt_batch_size', None))

            # Calculate system-level scores
            avg_meteor_scores = np.mean(list(meteor_results.values()), axis=1).tolist()
            avg_rouge_scores = np.mean(list(rouge_results.values()), axis=1).tolist()
            avg_bertscore_scores = np.mean(list(bertscore_results.values()), axis=1).tolist()
            avg_bleurt_scores = np.mean(list(bleurt_results.values()), axis=1).tolist()

            if not results:
                # Save the metric names as column names for the results table
                metric_names = ['BLEU (corpus)'] + list(meteor_results.keys()) + list(rouge_results.keys()) + list(bertscore_results.keys()) + list(bleurt_results.keys())
                results.append(metric_names)

            results.append([bleu_results['BLEU (corpus)']] + avg_meteor_scores + avg_rouge_scores + avg_bertscore_scores + avg_bleurt_scores)

            # Save segment-level scores to a CSV file
            if export_results:
                out_file_path = self.get_csv_export_path(suffix=f'_run{run}')
                df_out = pd.DataFrame({
                    'prediction': self.predictions,
                    'reference': self.references,
                    'BLEU': bleu_results['BLEU'],
                    **meteor_results,
                    **rouge_results,
                    **bertscore_results,
                    **bleurt_results
                })
                df_out.to_csv(out_file_path, index=False, encoding='utf-8-sig')

            # Generate new references (i.e., newly perturbated pseudo-utterances)
            if run < num_runs:
                self.generate_new_references()

        # Print system-level scores
        print('\n==== System-level scores ====')
        print('\t'.join(results[0]))
        for scores in results[1:]:
            print('\t'.join([str(round(score, 4)) for score in scores]))
        print()

    def calculate_bertscore_and_bleurt(self, num_runs: int = 1, shuffle_refs: Union[bool, str] = False,
                                       export_results: bool = False) -> None:
        results = []

        # Initialize scorers for metrics that depend on a model or a package, so they don't get loaded in every run
        bert_scorer1 = self.init_bert_scorer(model=BertScoreModelCheckpoint.DEBERTA_XLARGE_MNLI, batch_size=64, idf=True)
        bert_scorer2 = self.init_bert_scorer(model=BertScoreModelCheckpoint.DEBERTA_LARGE_MNLI, batch_size=64, idf=True)
        bleurt_scorer1 = self.init_bleurt_scorer(model=BleurtModelPath.BLEURT_20_D12)
        bleurt_scorer2 = self.init_bleurt_scorer(model=BleurtModelPath.BLEURT_20_D6)
        bleurt_scorer3 = self.init_bleurt_scorer(model=BleurtModelPath.BLEURT_20_D3)

        for run in tqdm(range(1, num_runs + 1), desc='Run'):
            # Shuffle references, if desired
            if shuffle_refs:
                self.shuffle_references(within_da_only=(shuffle_refs == 'da'))

            # Calculate segment-level scores
            bertscore_results1 = self.calculate_bertscore(bert_scorer1)
            bertscore_results2 = self.calculate_bertscore(bert_scorer2)
            bleurt_results1 = self.calculate_bleurt(bleurt_scorer1, batch_size=64)
            bleurt_results2 = self.calculate_bleurt(bleurt_scorer2, batch_size=64)
            bleurt_results3 = self.calculate_bleurt(bleurt_scorer3, batch_size=64)

            # Calculate system-level scores
            avg_bertscore_scores1 = np.mean(list(bertscore_results1.values()), axis=1).tolist()
            avg_bertscore_scores2 = np.mean(list(bertscore_results2.values()), axis=1).tolist()
            avg_bleurt_scores1 = np.mean(list(bleurt_results1.values()), axis=1).tolist()
            avg_bleurt_scores2 = np.mean(list(bleurt_results2.values()), axis=1).tolist()
            avg_bleurt_scores3 = np.mean(list(bleurt_results3.values()), axis=1).tolist()

            if not results:
                # Save the metric names as column names for the results table
                metric_names = list(bertscore_results1.keys()) + list(bertscore_results2.keys()) + list(bleurt_results1.keys()) + list(bleurt_results2.keys()) + list(bleurt_results3.keys())
                results.append(metric_names)

            results.append(avg_bertscore_scores1 + avg_bertscore_scores2 + avg_bleurt_scores1 + avg_bleurt_scores2 + avg_bleurt_scores3)

            # Save segment-level scores to a CSV file
            if export_results:
                out_file_path = self.get_csv_export_path(suffix=f'_neural_only_run{run}')
                df_out = pd.DataFrame({
                    'prediction': self.predictions,
                    'reference': self.references,
                    **bertscore_results1,
                    **bertscore_results2,
                    **bleurt_results1,
                    **bleurt_results2,
                    **bleurt_results3
                })
                df_out.to_csv(out_file_path, index=False, encoding='utf-8-sig')

            # Generate new references (i.e., newly perturbated pseudo-utterances)
            if run < num_runs:
                self.generate_new_references()

        # Print system-level scores
        print('\n==== System-level scores ====')
        print('\t'.join(results[0]))
        for scores in results[1:]:
            print('\t'.join([str(round(score, 4)) for score in scores]))
        print()

    def init_bert_scorer(self, model: Optional[str] = None, batch_size: int = 64,
                         idf: bool = False) -> bert_score.BERTScorer:
        # Default to the smallest model
        if not model:
            model = BertScoreModelCheckpoint.DEBERTA_LARGE_MNLI

        start = time.time()
        scorer = bert_score.BERTScorer(model_type=model, batch_size=batch_size, lang='en', idf=idf,
                                       idf_sents=self.references, rescale_with_baseline=True)
        print(f'>> BERTScore model load time: {round(time.time() - start, 2)} s')

        return scorer

    def calculate_bertscore(self, scorer: bert_score.BERTScorer) -> Dict[str, List[float]]:
        start = time.time()
        P, R, F1 = scorer.score(self.predictions, self.references)
        print(f'>> BERTScore execution time: {round(time.time() - start, 2)} s')

        # bert_score.plot_example(self.predictions[0], self.references[0], lang='en', rescale_with_baseline=True, idf=True)

        return {
            'BERTScore (precision)': [round(p, 4) for p in P.tolist()],
            'BERTScore (recall)': [round(r, 4) for r in R.tolist()],
            'BERTScore (F1)': [round(f1, 4) for f1 in F1.tolist()]
        }

    def calculate_bleu(self) -> Dict[str, List[float]]:
        bleu_scores = [round(corpus_bleu([pred], [[ref]]).score / 100, 4)
                       for pred, ref in zip(self.predictions, self.references)]
        bleu_system_level = corpus_bleu(self.predictions, [self.references]).score

        return {
            'BLEU': bleu_scores,
            'BLEU (corpus)': round(bleu_system_level / 100, 4)
        }

    def init_bleurt_scorer(self, model: Optional[str] = None) -> LengthBatchingBleurtScorer:
        # Default to the smallest model
        if not model:
            model = BleurtModelPath.BLEURT_20_D3

        # Initialize a scorer with length-based batching enabled, which significantly speeds up the computation
        start = time.time()
        scorer = LengthBatchingBleurtScorer(model)
        print(f'>> BLEURT model load time: {round(time.time() - start, 2)} s')

        return scorer

    def calculate_bleurt(self, scorer: LengthBatchingBleurtScorer,
                         batch_size: Optional[int] = None) -> Dict[str, List[float]]:
        """Calculates the BLEURT metric using a pretrained model.

        To download a BLEURT model checkpoint, use the "eval/BLEURT/download_model.sh" script.
        """

        # Initialize a scorer with length-based batching enabled, which significantly speeds up the computation
        # start = time.time()
        # scorer = LengthBatchingBleurtScorer(model)
        # print(f'>> BLEURT model load time: {round(time.time() - start, 2)} s')

        start = time.time()
        bleurt_scores = scorer.score(candidates=self.predictions, references=self.references, batch_size=batch_size)
        print(f'>> BLEURT execution time: {round(time.time() - start, 2)} s')

        return {
            'BLEURT': [round(score, 4) for score in bleurt_scores]
        }

    def init_meteor_scorer(self) -> Metric:
        start = time.time()
        scorer = load_metric('meteor')
        print(f'>> METEOR metric load time: {round(time.time() - start, 2)} s')

        return scorer

    def calculate_meteor(self, scorer: Metric) -> Dict[str, List[float]]:
        meteor_scores = [round(scorer.compute(predictions=[pred], references=[ref])['meteor'], 4)
                         for pred, ref in zip(self.predictions, self.references)]

        return {
            'METEOR': meteor_scores
        }

    def calculate_rouge(self, use_stemmer: bool = False) -> Dict[str, List[float]]:
        rouge_metric = load_metric('rouge')
        rouge_scores = rouge_metric.compute(predictions=self.predictions, references=self.references,
                                            rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=False,
                                            use_stemmer=use_stemmer)

        return {
            'ROUGE-1 (recall)': [round(score.recall, 4) for score in rouge_scores['rouge1']],
            'ROUGE-1 (F1)': [round(score.fmeasure, 4) for score in rouge_scores['rouge1']],
            'ROUGE-2 (recall)': [round(score.recall, 4) for score in rouge_scores['rouge2']],
            'ROUGE-2 (F1)': [round(score.fmeasure, 4) for score in rouge_scores['rouge2']],
            'ROUGE-L (recall)': [round(score.recall, 4) for score in rouge_scores['rougeL']],
            'ROUGE-L (F1)': [round(score.fmeasure, 4) for score in rouge_scores['rougeL']]
        }


if __name__ == '__main__':
    pseudo_ref_config = {
        'separator': ' ',
        'include_slot_names': False,
        'bool_slot_name_mode': SlotNameMode.SINGLE_WORD,
        'lowercase': False,
        # 'perturbation': PerturbationMode.DUPLICATE,
        # 'perturbation_n': 1,
    }
    metrics_config = {
        'bertscore_model': BertScoreModelCheckpoint.DEBERTA_LARGE_MNLI,
        'bertscore_batch_size': 64,
        'bertscore_idf': True,
        'bleurt_model': BleurtModelPath.BLEURT_20_D12,
        'bleurt_batch_size': 64,
        'rouge_use_stemmer': False,
    }

    evaluator = PseudoReferenceMetricEvaluator('video_game', partition='valid', **pseudo_ref_config)
    # evaluator = PseudoReferenceMetricEvaluator('rest_e2e_cleaned', partition='valid', **pseudo_ref_config)

    evaluator.calculate_all_metrics(num_runs=1, shuffle_refs=False, export_results=True, **metrics_config)
    # evaluator.calculate_all_metrics(num_runs=5, shuffle_refs=True, export_results=False, **metrics_config)

    # evaluator.calculate_bertscore_and_bleurt(num_runs=5, shuffle_refs=False, export_results=True)
    # evaluator.calculate_bertscore_and_bleurt(num_runs=5, shuffle_refs=True, export_results=False)
