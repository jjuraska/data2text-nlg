import sys


class TaskConfig(object):
    def __init__(self, config):
        self.pretrained_model = config.get('pretrained_model', 'gpt2')
        self.batch_size = config.get('batch_size', 1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.convert_slot_names = config.get('convert_slot_names', False)


class TrainingConfig(TaskConfig):
    def __init__(self, config):
        super().__init__(config)

        self.num_epochs = config.get('num_epochs', 1)
        self.num_warmup_steps = config.get('num_warmup_steps', 100)
        self.lr = config.get('lr', 5e-5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.eval_interval_in_steps = config.get('eval_interval_in_steps', 100)
        self.fp16 = config.get('fp16', False)


class TestConfig(TaskConfig):
    def __init__(self, config):
        super().__init__(config)

        if 'checkpoint_epoch' not in config or 'checkpoint_step' not in config:
            print('Error: checkpoint epoch or step not provided in the task configuration')
            sys.exit()

        self.checkpoint_epoch = config['checkpoint_epoch']
        self.checkpoint_step = config['checkpoint_step']
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
