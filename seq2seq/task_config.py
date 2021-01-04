import os
import re
import sys


class TaskConfig(object):
    def __init__(self, config):
        self.model_name = config.get('model_name')
        self.pretrained = config.get('pretrained', False)
        self.tokenizer_name = config.get('tokenizer_name')
        self.checkpoint_epoch = config.get('checkpoint_epoch')
        self.checkpoint_step = config.get('checkpoint_step')
        self.batch_size = config.get('batch_size', 1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.lowercase = config.get('lowercase', False)
        self.convert_slot_names = config.get('convert_slot_names', False)
        self.use_token_type_ids = config.get('use_token_type_ids', False)
        # if self.use_token_type_ids and not self.convert_slot_names:
        #     print('Error: the "use_token_type_ids" parameter can only be True when "convert_slot_names" is True.')
        #     sys.exit()


class TrainingConfig(TaskConfig):
    def __init__(self, config):
        super().__init__(config)

        self.num_slot_permutations = config.get('num_slot_permutations', 0)
        self.num_epochs = config.get('num_epochs', 1)
        self.num_warmup_steps = config.get('num_warmup_steps', 100)
        self.lr = config.get('lr', 5e-5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.eval_batch_size = config.get('eval_batch_size', self.batch_size)
        self.eval_times_per_epoch = config.get('eval_times_per_epoch', 1)
        self.eval_every_n_epochs = config.get('eval_every_n_epochs', 1)
        self.fp16 = config.get('fp16', False)


class TestConfig(TaskConfig):
    def __init__(self, config):
        super().__init__(config)

        if self.checkpoint_epoch is None or self.checkpoint_step is None:
            self.extract_epoch_and_step_from_model_path()

        self.num_beams = config.get('num_beams', 1)
        self.early_stopping = config.get('beam_search_early_stopping', False)
        self.do_sample = config.get('do_sample', False)
        self.top_p = config.get('top_p', 1.0)
        self.top_k = config.get('top_k', 0)
        self.temperature = config.get('temperature', 1.0)
        self.no_repeat_ngram_size = config.get('no_repeat_ngram_size', 0)
        self.repetition_penalty = config.get('repetition_penalty', 1.0)
        self.length_penalty = config.get('length_penalty', 1.0)
        self.num_return_sequences = config.get('num_return_sequences', self.num_beams)
        self.semantic_reranking = config.get('semantic_reranking', False)

    def extract_epoch_and_step_from_model_path(self):
        epoch_and_step_str = os.path.split(self.model_name.rstrip(r'\/'))[-1]
        try:
            epoch, step = re.search(r'epoch_(\d+)_step_(\d+)', epoch_and_step_str).groups()
            self.checkpoint_epoch = int(epoch)
            self.checkpoint_step = int(step)
        except AttributeError:
            print('Error: checkpoint epoch or step neither provided in the task configuration, nor found in the model name.')
            sys.exit()
