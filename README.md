# SeA-GuiDe

This repository contains the implementation of the Semantically Attention-Guided Decoding (SeA-GuiDe) method introduced in the [Attention Is Indeed All You Need: Semantically Attention-Guided Decoding for Data-to-Text NLG (Juraska & Walker, 2021)](https://aclanthology.org/2021.inlg-1.45/) paper. This decoding method makes a better use of the cross-attention component of the already complex and enormous pretrained generative language models (LMs) to achieve significantly higher semantic accuracy for data-to-text NLG, while preserving the otherwise high quality of the output text. It is an automatic method, exploiting information already present in the model, but in an interpretable way. SeA-GuiDe requires no training, annotation, data augmentation, or model modifications, and can thus be effortlessly used with different models and domains.

Since SeA-GuiDe is intended for use with encoder-decoder models with cross-attention, this repository offers a simple API to fine-tune [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html), [T5](https://huggingface.co/transformers/model_doc/t5.html), or [BART](https://huggingface.co/transformers/model_doc/bart.html) (using the PyTorch implementations in Hugging Face's [transformers](https://huggingface.co/transformers/index.html) library), and subsequently run inference with a fine-tuned model using greedy search, beam search, or SeA-GuiDe as the decoding method.

## Fine-Tuning a Pretrained Model

As an example, here is a command to execute fine-tuning of the T5-small model on the ViGGO dataset:

```
python -m seq2seq.run_task -t train -c t5-small -d video_game
```

Currently supported values for the domain/dataset argument (`-d`) are: `video_game` (ViGGO), `rest_e2e` (E2E), `rest_e2e_cleaned` (Cleaned E2E), `multiwoz` (MultiWOZ 2.1).

The `-c` argument specifies the name of the training configuration to be used. The value must match the name (without the extension) of a YAML config file in the corresponding dataset directory under `seq2seq/config`. For example, the above command would result in the use of the `seq2seq/config/video_game/train/t5-small.yaml` config file for the fine-tuning task. Below is an example config file:

```
model_name: "t5-small"
pretrained: True
lowercase: False
num_epochs: 20
batch_size: 32
eval_batch_size: 256
max_seq_length: 128
num_warmup_steps: 100
lr: 2.0e-4
max_grad_norm: 1.0
eval_times_per_epoch: 1
fp16: True
```

During the fine-tuning, model checkpoints are saved in the `seq2seq/model` directory.

## Evaluating a Fine-Tuned Model

To evaluate a fine-tuned model on a dataset's test partition, the task argument (`-t`) changes to `test`, and a test configuration (different from a training configuration) must be provided, e.g.:

```
python -m seq2seq.run_task -t test -c t5-small_beam_search -d video_game
```

The config file used in the above command would be `seq2seq/config/video_game/test/t5-small_beam_search.yaml`, which can look something like this:

```
model_name: "seq2seq/export/video_game/t5-small_lr_2e-4_bs_32_wus_100/epoch_16_step_160"
pretrained: True
lowercase: False
batch_size: 32
max_seq_length: 128
num_beams: 10
beam_search_early_stopping: True
do_sample: False
length_penalty: 1.0
semantic_decoding: True
semantic_reranking: True
```

Note that the example config file points to a different model checkpoint path. Since the `seq2seq/model` directory gets overwritten whenever a new fine-tuning experiment is executed, you'll want to move the best checkpoint to a different directory for evaluation and for long-term storage in general.

Setting both the `semantic_decoding` and the `semantic_reranking` parameter to True enables SeA-GuiDe. Setting them both to False results in vanilla beam search decoding. Setting only `semantic_reranking` to True enables the heuristic slot aligner-based reranking (see the paper for details). Lastly, in order to use greedy decoding, change the `num_beams` parameter to 1.
