import copy
import numpy as np
import os
import pickle
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers.generation_logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

from beam_search_scoring import SemanticBeamSearchScorer
import model_utils as model_utils
from semantic_tracking import (
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
                        model,
                        attention_mask=mask_tensor,
                        max_length=config.max_seq_length,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        beam_size=config.num_beams,
                        length_penalty=config.length_penalty,
                        do_sample=config.do_sample,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        early_stopping=config.early_stopping,
                        output_attentions=False,
                        device=device)
                else:
                    num_seqs_per_input = config.num_return_sequences if config.do_sample else 1
                    outputs, attn_weights, slot_error_list = semantic_decoding(
                        input_tensor,
                        batch['slot_spans'],
                        model,
                        attention_mask=mask_tensor,
                        max_length=config.max_seq_length,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=config.do_sample,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        num_return_sequences=config.num_return_sequences,
                        output_attentions=False,
                        device=device)

                slot_errors.extend(slot_error_list)

                # # Save the input and output sequences (as lists of tokens) along with the attention weights
                # attn_weights['input_tokens'] = [tokenizer.decode(input_id, skip_special_tokens=False)
                #                                 for input_id in batch['input_ids'][0]]
                # attn_weights['output_tokens'] = [tokenizer.decode(output_id, skip_special_tokens=False)
                #                                  for output_id in outputs[0][1:]]
                #
                # # Export attention weights for visualization
                # with open(os.path.join('attention', 'attention_weights.pkl'), 'wb') as f_attn:
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
                                         temperature=config.temperature,
                                         top_p=config.top_p,
                                         top_k=config.top_k,
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


@torch.no_grad()
def semantic_decoding(input_ids, batch_slot_spans, model, attention_mask=None, max_length=128, pad_token_id=None,
                      bos_token_id=None, eos_token_id=None, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                      num_return_sequences=1, output_attentions=False, device='cpu'):
    """Performs a semantically attention-guided inference from a structured MR input using greedy search."""
    outputs = None
    past = None
    attention_weights = {}
    batch_size = input_ids.size(0)

    if not do_sample and num_return_sequences > 1:
        raise ValueError(f'`num_return_sequences` must be 1 when using greedy search decoding, but is {num_return_sequences}')

    # logits_processor = LogitsProcessorList([
    #     NoRepeatNGramLogitsProcessor(model.config.no_repeat_ngram_size),
    # ])
    logits_processor = LogitsProcessorList()

    stopping_criteria = StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length)
    ])

    logits_warper = LogitsProcessorList()
    if do_sample:
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k != 0:
            logits_warper.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # Initialize the decoder's input sequence with the corresponding token
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * batch_size, dtype=torch.long).to(device)

    # Run the input sequence through the encoder and get the encoded input sequence
    encoder = model.get_encoder()
    encoded_sequence = encoder(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)

    # Determine the indices of special tokens in the input sequence
    special_tokens = torch.tensor(
        [tok for tok in [bos_token_id, eos_token_id] if tok is not None], device=input_ids.device)
    special_token_idxs = torch.nonzero(
        input_ids.detach().repeat_interleave(num_return_sequences, dim=0)[:, :, None] == special_tokens, as_tuple=True)[:-1]

    # DEBUG
    # print('>> Indices of special tokens:')
    # print(special_token_idxs)
    # print()

    # For each candidate in the beam prepare its own slot tracking dict by duplicating the initial one
    batch_slot_spans = [copy.deepcopy(slot_spans) for slot_spans in batch_slot_spans for _ in range(num_return_sequences)]

    if output_attentions:
        # Save the encoder's self-attention weights
        attention_weights['enc_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in encoded_sequence.attentions
        ]

    if num_return_sequences > 1:
        decoder_input_ids, encoded_sequence, attention_mask = expand_inputs_for_generation(
            decoder_input_ids, encoded_sequence, attention_mask=attention_mask, expand_size=num_return_sequences)

    # Keep track of which sequences are not yet finished
    unfinished_sequences = decoder_input_ids.new(decoder_input_ids.shape[0]).fill_(1)

    for step in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(decoder_input_ids, past=past, attention_mask=attention_mask,
                                                           use_cache=True, encoder_outputs=encoded_sequence)

        # Reuse the encoded inputs, and pass the sequence generated so far as inputs to the decoder
        outputs = model(**model_inputs, output_attentions=True, return_dict=True)

        logits = outputs.logits

        track_slot_mentions(
            logits,
            outputs.cross_attentions,
            batch_slot_spans,
            eos_token_id,
            special_token_idxs
        )

        next_token_scores = logits_processor(decoder_input_ids, logits[:, -1, :])
        next_token_scores = logits_warper(decoder_input_ids, next_token_scores)

        if do_sample:
            # Sample one token based on the probability distribution
            token_probs = F.softmax(next_token_scores, dim=-1)
            next_decoder_input_ids = torch.multinomial(token_probs, num_samples=1).squeeze(1)
        else:
            # Select the token with the highest probability as the next generated token (~ greedy decoding)
            next_decoder_input_ids = torch.argmax(next_token_scores, dim=-1)

        # Sets the next token of finished sequences to be the padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_decoder_input_ids = next_decoder_input_ids * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # Append the current output token's ID to the sequence generated so far
        decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids.unsqueeze(-1)], dim=-1)

        past = update_past_for_next_time_step(outputs)

        # Mark a sequence finished if EOS token was determined as the next token
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_decoder_input_ids != eos_token_id).long())

        # DEBUG
        # for i in range(len(outputs.cross_attentions)):
        #     print(outputs.cross_attentions[i].size())
        # print()

        # Terminate as soon as each sequence is finished or the maximum sequence length has been reached
        if unfinished_sequences.max() == 0 or stopping_criteria(decoder_input_ids, None):
            break

    # Add a dimension for compatibility with beam search
    batch_slot_spans_final = [batch_slot_spans[beam_start_idx:beam_start_idx + num_return_sequences]
                              for beam_start_idx in range(0, len(batch_slot_spans), num_return_sequences)]

    if output_attentions and outputs:
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


@torch.no_grad()
def semantic_decoding_beam_search(input_ids, batch_slot_spans, model, attention_mask=None, max_length=128,
                                  pad_token_id=None, bos_token_id=None, eos_token_id=None, beam_size=1,
                                  length_penalty=1.0, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                                  early_stopping=False, output_attentions=False, device='cpu'):
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

    logits_warper = LogitsProcessorList()
    if do_sample:
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k != 0:
            logits_warper.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if beam_size > 1 else 1)))
        if top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if beam_size > 1 else 1)))

    # Initialize the decoder's input sequence with the corresponding token
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * batch_size, dtype=torch.long).to(device)

    # Run the input sequence through the encoder and get the encoded input sequence
    encoder = model.get_encoder()
    encoded_sequence = encoder(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)

    # Determine the indices of special tokens in the input sequence
    special_tokens = torch.tensor(
        [tok for tok in [bos_token_id, eos_token_id] if tok is not None], device=input_ids.device)
    special_token_idxs = torch.nonzero(
        input_ids.detach().repeat_interleave(beam_size, dim=0)[:, :, None] == special_tokens, as_tuple=True)[:-1]

    # DEBUG
    # print('>> Indices of special tokens:')
    # print(special_token_idxs)
    # print()

    # For each candidate in the beam prepare its own slot tracking dict by duplicating the initial one
    batch_slot_spans = [copy.deepcopy(slot_spans) for slot_spans in batch_slot_spans for _ in range(beam_size)]

    if output_attentions:
        # Save the encoder's self-attention weights
        attention_weights['enc_attn_weights'] = [
            weight_tensor.squeeze().tolist() for weight_tensor in encoded_sequence.attentions
        ]

    # Initialize the beam search scorer
    beam_scorer = SemanticBeamSearchScorer(
        batch_size=batch_size,
        num_beams=beam_size,
        device=device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=beam_size,
    )

    decoder_input_ids, encoded_sequence, attention_mask = expand_inputs_for_generation(
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
        next_token_scores = logits_warper(decoder_input_ids, next_token_scores)

        # Reshape the token score tensor for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

        if do_sample:
            # Sample (2 * beam_size) partial sequences based on their probability distribution
            token_probs = F.softmax(next_token_scores, dim=-1)
            best_next_token_ids = torch.multinomial(token_probs, num_samples=2 * beam_size)
            best_next_token_scores = torch.gather(next_token_scores, -1, best_next_token_ids)

            # Sort the sampled tokens by their probability
            best_next_token_scores, sorted_indices = torch.sort(best_next_token_scores, descending=True, dim=1)
            best_next_token_ids = torch.gather(best_next_token_ids, -1, sorted_indices)
        else:
            # Keep (2 * beam_size) partial sequences with the highest scores
            best_next_token_scores, best_next_token_ids = torch.topk(
                next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

        best_beam_idxs = (best_next_token_ids / vocab_size).long()
        best_next_token_ids = best_next_token_ids % vocab_size

        track_slot_mentions(
            logits,
            outputs.cross_attentions,
            batch_slot_spans,
            eos_token_id,
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
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs['next_beam_scores']
        beam_idxs = beam_outputs['next_beam_indices']
        beam_next_tokens = beam_outputs['next_beam_tokens']

        # Append the current output token's ID to the sequence generated so far
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idxs, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        past = update_past_for_next_time_step(outputs)
        if past is not None and beam_idxs is not None:
            past = model._reorder_cache(past, beam_idxs)

        batch_slot_spans = rearrange_slot_mentions_for_next_time_step(batch_slot_spans, beam_idxs)

        if beam_scorer.is_done or stopping_criteria(decoder_input_ids, None):
            break

    sequence_outputs = beam_scorer.finalize(decoder_input_ids,
                                            beam_scores,
                                            best_next_token_ids,
                                            best_beam_idxs,
                                            max_length,
                                            batch_slot_spans,
                                            pad_token_id=pad_token_id,
                                            eos_token_id=eos_token_id)

    slot_mentions_final = sequence_outputs['sequence_slot_mentions']
    batch_slot_spans_final = [slot_mentions_final[beam_start_idx:beam_start_idx + beam_size]
                              for beam_start_idx in range(0, len(slot_mentions_final), beam_size)]

    if output_attentions and outputs:
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


def expand_inputs_for_generation(input_ids, enc_outputs, attention_mask=None, expand_size=1):
    """Borrowed from Huggingface's transformers library."""

    expanded_indices = torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    input_ids = input_ids.index_select(0, expanded_indices)

    if attention_mask is not None:
        attention_mask = attention_mask.index_select(0, expanded_indices)

    enc_outputs['last_hidden_state'] = enc_outputs.last_hidden_state.index_select(
        0, expanded_indices.to(enc_outputs.last_hidden_state.device)
    )

    return input_ids, enc_outputs, attention_mask


def update_past_for_next_time_step(outputs):
    if 'past_key_values' in outputs:
        past = outputs.past_key_values
    elif 'mems' in outputs:
        past = outputs.mems
    elif 'past_buckets_states' in outputs:
        past = outputs.past_buckets_states
    else:
        past = None

    return past
