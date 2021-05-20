import copy
import numpy as np
import os
import pickle
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers.generation_logits_process import LogitsProcessorList, NoRepeatNGramLogitsProcessor
from transformers.generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

from seq2seq.beam_search_scoring import SemanticBeamSearchScorer
import seq2seq.model_utils as model_utils
from seq2seq.semantic_tracking import (
    evaluate_slot_mentions,
    rearrange_slot_mentions_for_next_time_step,
    track_slot_mentions
)


def generate_and_decode(config, data_loader, tokenizer, model, is_enc_dec, is_validation=False, bool_slots=None,
                        device='cpu'):
    generated_sequences = []
    slot_errors = []

    if 'gpt2' in config.model_name:
        # Set the tokenizer to padding input sequences on the left side in order to enable batch inference
        tokenizer.padding_side = 'left'

    for batch in tqdm(data_loader, desc='Generating'):
        batch = model_utils.prepare_batch(config, batch, tokenizer, is_enc_dec, include_labels=False,
                                          bool_slots=bool_slots)

        input_tensor = batch['input_ids'].to(device)
        mask_tensor = batch['attention_mask'].to(device)

        model_specific_args = {}
        if batch.get('token_type_ids') is not None:
            model_specific_args['token_type_ids'] = batch['token_type_ids'].to(device)
            if hasattr(config, 'num_return_sequences'):
                model_specific_args['expand_token_type_size'] = config.num_return_sequences

        # DEBUG
        # print()
        # print('>> ENCODED inputs:', inputs['input_ids'])
        # print('>> DECODED inputs:\n', tokenizer.decode(inputs['input_ids'][0]))
        # print()
        # print('>> ENCODED input mask:', inputs['attention_mask'])
        # print()

        if is_validation:
            num_seqs_per_input = 1
            outputs = model.generate(input_tensor,
                                     attention_mask=mask_tensor,
                                     min_length=1,
                                     max_length=config.max_seq_length,
                                     **model_specific_args)
        else:
            if config.semantic_decoding:
                if config.num_beams > 1:
                    num_seqs_per_input = config.num_return_sequences
                    outputs, attn_weights, slot_error_list = semantic_decoding_beam_search(
                        input_tensor,
                        batch['slot_spans'],
                        tokenizer,
                        model,
                        attention_mask=mask_tensor,
                        max_length=config.max_seq_length,
                        beam_size=config.num_beams,
                        length_penalty=config.length_penalty,
                        early_stopping=config.early_stopping,
                        device=device)
                else:
                    num_seqs_per_input = 1
                    outputs, attn_weights, slot_error_list = semantic_decoding(input_tensor,
                                                                               batch['slot_spans'],
                                                                               tokenizer,
                                                                               model,
                                                                               max_length=config.max_seq_length,
                                                                               device=device)

                slot_errors.extend(slot_error_list)

                # # Save the input and output sequences (as lists of tokens) along with the attention weights
                # attn_weights['input_tokens'] = [tokenizer.decode(input_id, skip_special_tokens=False)
                #                                 for input_id in batch['input_ids'][0]]
                # attn_weights['output_tokens'] = [tokenizer.decode(output_id, skip_special_tokens=False)
                #                                  for output_id in outputs[0][1:]]
                #
                # # Export attention weights for visualization
                # with open(os.path.join('seq2seq', 'attention', 'attention_weights.pkl'), 'wb') as f_attn:
                #     pickle.dump(attn_weights, f_attn)
            else:
                num_seqs_per_input = config.num_return_sequences
                outputs = model.generate(input_tensor,
                                         attention_mask=mask_tensor,
                                         min_length=1,
                                         max_length=config.max_seq_length,
                                         num_beams=config.num_beams,
                                         early_stopping=config.early_stopping,
                                         no_repeat_ngram_size=config.no_repeat_ngram_size,
                                         do_sample=config.do_sample,
                                         top_p=config.top_p,
                                         top_k=config.top_k,
                                         temperature=config.temperature,
                                         repetition_penalty=config.repetition_penalty,
                                         length_penalty=config.length_penalty,
                                         num_return_sequences=config.num_return_sequences,
                                         **model_specific_args)

        generated_sequences.extend(decode_model_outputs(outputs, num_seqs_per_input, tokenizer, is_enc_dec))

    if 'gpt2' in config.model_name:
        # Set the tokenizer back to padding input sequences on the right
        tokenizer.padding_side = 'right'

    return generated_sequences, slot_errors


def decode_model_outputs(sequences, num_seqs_per_input, tokenizer, is_enc_dec):
    outputs_decoded = []

    for beam_idx in range(0, len(sequences), num_seqs_per_input):
        beam_sequences = sequences[beam_idx:beam_idx + num_seqs_per_input]
        beam_decoded = []

        for seq in beam_sequences:
            if is_enc_dec:
                utt_decoded = tokenizer.decode(seq, skip_special_tokens=True)
            else:
                utt_beg_pos = np.where(seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
                utt_decoded = tokenizer.decode(seq[utt_beg_pos:], skip_special_tokens=True)

            beam_decoded.append(utt_decoded)

        outputs_decoded.append(beam_decoded)

    return outputs_decoded


def semantic_decoding(input_ids, batch_slot_spans, tokenizer, model, max_length=128, device='cpu'):
    """Performs a semantically attention-guided inference from a structured MR input using greedy search."""
    outputs = None
    past = None
    attention_weights = {}

    # logits_processor = LogitsProcessorList([
    #     NoRepeatNGramLogitsProcessor(model.config.no_repeat_ngram_size),
    # ])
    logits_processor = LogitsProcessorList()

    stopping_criteria = StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length)
    ])

    # Initialize the decoder's input sequence with the corresponding token
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long).to(device)

    # Run the input sequence through the encoder and get the encoded input sequence
    encoder = model.get_encoder()
    encoded_sequence = encoder(input_ids, output_attentions=True)

    # Determine the indices of special tokens in the input sequence
    special_tokens = torch.tensor(
        [tok for tok in [tokenizer.bos_token_id, tokenizer.eos_token_id] if tok is not None], device=input_ids.device)
    special_token_idxs = torch.nonzero(input_ids.detach()[:, :, None] == special_tokens, as_tuple=True)[:-1]

    # DEBUG
    # print('>> Indices of special tokens:')
    # print(special_token_idxs)
    # print()

    # Save the encoder's self-attention weights
    attention_weights['enc_attn_weights'] = [
        weight_tensor.squeeze().tolist() for weight_tensor in encoded_sequence.attentions
    ]

    for step in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(decoder_input_ids, past=past, attention_mask=None,
                                                           use_cache=True, encoder_outputs=encoded_sequence)

        # Reuse the encoded inputs, and pass the sequence generated so far as inputs to the decoder
        outputs = model(**model_inputs, output_attentions=True, return_dict=True)

        logits = outputs.logits

        track_slot_mentions(
            logits,
            outputs.cross_attentions,
            batch_slot_spans,
            tokenizer.eos_token_id,
            special_token_idxs
        )

        next_token_scores = logits_processor(decoder_input_ids, logits[:, -1, :])

        # Select the token with the highest probability as the next generated token (~ greedy decoding)
        next_decoder_input_ids = torch.argmax(next_token_scores, dim=-1)

        # Append the current output token's ID to the sequence generated so far
        decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids.unsqueeze(-1)], dim=-1)

        # DEBUG
        # for i in range(len(outputs.cross_attentions)):
        #     print(outputs.cross_attentions[i].size())
        # print()

        # Terminate as soon as the decoder generates the EOS token
        if next_decoder_input_ids.item() == tokenizer.eos_token_id or stopping_criteria(decoder_input_ids, None):
            break

    # Add a dimension for compatibility with beam search
    batch_slot_spans_final = [[slot_spans] for slot_spans in batch_slot_spans]

    if outputs:
        # Save the decoder's self- and cross-attention weights; shape = (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        attention_weights['dec_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in outputs.decoder_attentions
        ]
        attention_weights['cross_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in outputs.cross_attentions
        ]

    # DEBUG
    # print('>> Slot mentions:')
    # print(batch_slot_spans_final)
    # print()

    slot_errors = evaluate_slot_mentions(batch_slot_spans_final)

    # DEBUG
    # print('>> Slot errors:')
    # print(slot_errors)
    # print()

    return decoder_input_ids, attention_weights, slot_errors


def semantic_decoding_beam_search(input_ids, batch_slot_spans, tokenizer, model, attention_mask=None, max_length=128,
                                  beam_size=1, length_penalty=1.0, early_stopping=False, device='cpu'):
    """Performs a semantically attention-guided inference from a structured MR input using beam search."""
    outputs = None
    past = None
    attention_weights = {}
    batch_size = input_ids.size(0)

    # logits_processor = LogitsProcessorList([
    #     NoRepeatNGramLogitsProcessor(model.config.no_repeat_ngram_size),
    # ])
    logits_processor = LogitsProcessorList()

    stopping_criteria = StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length),
    ])

    # Initialize the decoder's input sequence with the corresponding token
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * batch_size, dtype=torch.long).to(device)

    # Run the input sequence through the encoder and get the encoded input sequence
    encoder = model.get_encoder()
    encoded_sequence = encoder(input_ids, attention_mask=attention_mask, output_attentions=True)

    # Determine the indices of special tokens in the input sequence
    special_tokens = torch.tensor(
        [tok for tok in [tokenizer.bos_token_id, tokenizer.eos_token_id] if tok is not None], device=input_ids.device)
    special_token_idxs = torch.nonzero(
        input_ids.detach().repeat_interleave(beam_size, dim=0)[:, :, None] == special_tokens, as_tuple=True)[:-1]

    # DEBUG
    # print('>> Indices of special tokens:')
    # print(special_token_idxs)
    # print()

    # For each candidate in the beam prepare its own slot tracking dict by duplicating the initial one
    batch_slot_spans = [copy.deepcopy(slot_spans) for slot_spans in batch_slot_spans for _ in range(beam_size)]

    # Save the encoder's self-attention weights
    attention_weights['enc_attn_weights'] = [
        weight_tensor.squeeze().tolist() for weight_tensor in encoded_sequence.attentions
    ]

    # Initialize the beam search scorer
    beam_scorer = SemanticBeamSearchScorer(
        batch_size=batch_size,
        max_length=max_length,
        num_beams=beam_size,
        device=device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=beam_size,
    )

    decoder_input_ids, encoded_sequence, attention_mask = expand_inputs_for_beam_search(
        decoder_input_ids, encoded_sequence, attention_mask=attention_mask, expand_size=beam_size)

    # Prepare a tensor for storing beam scores at each decoder time step
    beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * beam_size,))

    for step in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(decoder_input_ids, past=past, attention_mask=attention_mask,
                                                           use_cache=True, encoder_outputs=encoded_sequence)

        # Reuse the encoded inputs, and pass the sequence generated so far as inputs to the decoder
        outputs = model(**model_inputs, output_attentions=True, return_dict=True)

        logits = outputs.logits

        # Add the corresponding partial sequence log-probabilities to those of the next tokens
        next_token_scores = F.log_softmax(logits[:, -1, :], dim=-1)  # (batch_size * beam_size, vocab_size)
        next_token_scores = logits_processor(decoder_input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # Reshape the token score tensor for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

        # Keep (2 * beam_size) partial sequences with the highest scores
        best_next_token_scores, best_next_token_ids = torch.topk(
            next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
        best_beam_idxs = best_next_token_ids // vocab_size
        best_next_token_ids = best_next_token_ids % vocab_size

        track_slot_mentions(
            logits,
            outputs.cross_attentions,
            batch_slot_spans,
            tokenizer.eos_token_id,
            special_token_idxs,
            num_seqs_per_input=beam_size
        )

        # Extend the current beam hypotheses with the next tokens
        beam_outputs = beam_scorer.process(
            decoder_input_ids,
            best_next_token_scores,
            best_next_token_ids,
            best_beam_idxs,
            batch_slot_spans,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        beam_scores = beam_outputs['next_beam_scores']
        beam_idxs = beam_outputs['next_beam_indices']
        beam_next_tokens = beam_outputs['next_beam_tokens']

        # Append the current output token's ID to the sequence generated so far
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idxs, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        past = update_past_for_next_time_step(outputs, model, beam_idxs)
        batch_slot_spans = rearrange_slot_mentions_for_next_time_step(batch_slot_spans, beam_idxs)

        if beam_scorer.is_done or stopping_criteria(decoder_input_ids, None):
            break

    sequence_outputs = beam_scorer.finalize(decoder_input_ids,
                                            beam_scores,
                                            best_next_token_ids,
                                            best_beam_idxs,
                                            # max_length,
                                            batch_slot_spans,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id)

    slot_mentions_final = sequence_outputs['sequence_slot_mentions']
    batch_slot_spans_final = [slot_mentions_final[beam_start_idx:beam_start_idx + beam_size]
                              for beam_start_idx in range(0, len(slot_mentions_final), beam_size)]

    if outputs:
        # Save the decoder's self- and cross-attention weights; shape = (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        attention_weights['dec_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in outputs.decoder_attentions
        ]
        attention_weights['cross_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in outputs.cross_attentions
        ]

    # DEBUG
    # print('>> Slot mentions:')
    # print(batch_slot_spans_final)
    # print()

    slot_errors = evaluate_slot_mentions(batch_slot_spans_final)

    # DEBUG
    # print('>> Slot errors:')
    # print(slot_errors)
    # print()

    return sequence_outputs['sequences'], attention_weights, slot_errors


def expand_inputs_for_beam_search(input_ids, enc_outputs, attention_mask=None, expand_size=1):
    """Borrowed from Huggingface's transformers library."""

    expanded_indices = torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    input_ids = input_ids.index_select(0, expanded_indices)

    if attention_mask is not None:
        attention_mask = attention_mask.index_select(0, expanded_indices)

    enc_outputs['last_hidden_state'] = enc_outputs.last_hidden_state.index_select(
        0, expanded_indices.to(enc_outputs.last_hidden_state.device)
    )

    return input_ids, enc_outputs, attention_mask


def update_past_for_next_time_step(outputs, model, beam_idxs):
    if 'past_key_values' in outputs:
        past = outputs.past_key_values
    elif 'mems' in outputs:
        past = outputs.mems
    elif 'past_buckets_states' in outputs:
        past = outputs.past_buckets_states
    else:
        past = None

    if past is not None:
        past = model._reorder_cache(past, beam_idxs)

    return past
