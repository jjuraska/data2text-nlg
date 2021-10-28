import copy
import torch


def track_slot_mentions(logits, attn_weights, slot_spans, eos_token_id, special_token_idxs, num_seqs_per_input=1):
    # Convert from a tuple to an array, and move the batch dimension to the front
    attn_weights = torch.stack(attn_weights, dim=1)
    # Ignore weights from other than the most recent step in the sequence
    attn_weights = attn_weights[:, :, :, -1, :].detach()  # (batch_size * beam_size, layers, heads, input_len)

    # DEBUG
    # print('>> logits.size():', logits.size())
    # print('>> attn_weights.size():', attn_weights.size())
    # print('>> special_token_idxs[0].shape:', special_token_idxs[0].shape)
    # print('>> special_token_idxs:', special_token_idxs)
    # print()

    most_probable_next_tokens = torch.argmax(logits[:, -1, :].detach(), axis=-1)

    # Set the special-token attention weights to zero
    if special_token_idxs:
        attn_weights[special_token_idxs[0], :, :, special_token_idxs[-1]] = 0.0

    # Extract the attention weights from the 1st decoder layer only, and aggregate them across heads
    attn_weights_first_layer = preprocess_attn_weights(
        attn_weights[:, 0:1, ...].clone(), head_agg_mode='max', layer_agg_mode=None)
    attn_weights_first_layer = binarize_weights(
        attn_weights_first_layer, threshold=0.9, keep_max_only=True).squeeze(2).squeeze(1)
    attn_idxs = torch.nonzero(attn_weights_first_layer == 1, as_tuple=True)

    # Update slot mentions with a high confidence
    update_slot_mentions(slot_spans, attn_idxs, confidence=True)

    batch_idxs_without_eos = torch.nonzero(most_probable_next_tokens != eos_token_id, as_tuple=True)
    attn_weights_agg = attn_weights.clone()
    attn_weights_agg[batch_idxs_without_eos] = 0.0

    # Remove slot mentions if they have a high attention weight associated with the EOS token
    attn_weights_agg = preprocess_attn_weights(attn_weights_agg, head_agg_mode='max', layer_agg_mode='avg')
    attn_weights_agg = binarize_weights(attn_weights_agg, threshold=0.1).squeeze(2).squeeze(1)
    attn_idxs = torch.nonzero(attn_weights_agg == 1, as_tuple=True)

    remove_slot_mentions(slot_spans, attn_idxs)

    batch_idxs_with_eos = torch.nonzero(most_probable_next_tokens == eos_token_id, as_tuple=True)
    attn_weights_agg = attn_weights.clone()
    attn_weights_agg[batch_idxs_with_eos] = 0.0

    num_layers = attn_weights.shape[1]
    middle_layer_idx = num_layers // 2

    # Aggregate the attention weights across both the heads and the layers
    attn_weights_agg = preprocess_attn_weights(
        attn_weights_agg[:, 0:middle_layer_idx, ...], head_agg_mode='max', layer_agg_mode='avg')
    attn_weights_agg = binarize_weights(attn_weights_agg, threshold=0.3, keep_max_only=False).squeeze(2).squeeze(1)
    attn_idxs = torch.nonzero(attn_weights_agg == 1, as_tuple=True)

    # DEBUG
    # print('>> attn_weights_agg.shape (after binarizing):', attn_weights_agg.shape)
    # print('>> attn_idxs:', attn_idxs)
    # print()

    # Update slot mentions with a low confidence
    update_slot_mentions(slot_spans, attn_idxs, confidence=False)


def preprocess_attn_weights(attn_weights, head_agg_mode=None, layer_agg_mode=None, threshold=0.0):
    if head_agg_mode:
        # num_heads = attn_weights.shape[1]
        attn_weights = aggregate_across_heads(attn_weights, mode=head_agg_mode)

    if layer_agg_mode:
        # num_layers = attn_weights.shape[0]
        # middle_layer_idx = num_layers // 2

        attn_weights = aggregate_across_layers(attn_weights, mode=layer_agg_mode)
        # attn_weights = aggregate_across_layers(attn_weights[0:middle_layer_idx, :, :], mode=layer_agg_mode)
        # for layer_idx in range(1, middle_layer_idx + 1):
        #     attn_weights_aggr = aggregate_across_layers(attn_weights[0:layer_idx, :, :], mode=layer_agg_mode)
        #     max_weights = attn_weights_aggr.max(axis=-1)[:, :, np.newaxis]
        #     attn_weights_aggr[np.nonzero(attn_weights_aggr < threshold)] = 0
        #     if (attn_weights_aggr == max_weights).any() or layer_idx == middle_layer_idx:
        #         attn_weights = attn_weights_aggr
        #         break

    return attn_weights


def update_slot_mentions(slot_mention_batch, attn_idxs, confidence=False):
    for batch_idx, attn_idx in zip(attn_idxs[0].tolist(), attn_idxs[-1].tolist()):
        for slot in slot_mention_batch[batch_idx]:
            if all(slot['mentioned']) and all(slot['confidence']):
                continue

            attn_weight_matched = False

            if 'value_span' in slot and not slot['is_boolean']:
                # Check if attn_idx is between the position of the 1st element and the position of the last element
                if slot['value_span'][0][0] <= attn_idx <= slot['value_span'][-1][1]:
                    for elem_idx, value_elem_span in enumerate(slot['value_span']):
                        if value_elem_span[0] <= attn_idx <= value_elem_span[1]:
                            slot['mentioned'][elem_idx] = True
                            if not slot['confidence'][elem_idx]:
                                slot['confidence'][elem_idx] = confidence
                            attn_weight_matched = True
                            break
            else:
                # For Boolean slots and slots without a value, match the slot's name
                if slot['name_span'][0] <= attn_idx <= slot['name_span'][1]:
                    slot['mentioned'][0] = True
                    if not slot['confidence'][0]:
                        slot['confidence'][0] = confidence
                    attn_weight_matched = True

            if attn_weight_matched:
                break


def update_slot_mentions_ALT(slot_mention_batch, attn_idxs, confidence=False):
    for batch_idx, attn_idx in zip(attn_idxs[0], attn_idxs[-1]):
        for slot in slot_mention_batch[batch_idx]:
            if all(slot['mentioned']) and all(slot['confidence']):
                continue

            attn_weight_matched = False

            if 'value_span' in slot and (not slot['is_boolean'] or slot.get('is_boolean_neg', False)):
                for elem_idx, value_elem_span in enumerate(slot['value_span']):
                    # TODO: optimize by breaking out of the loop if attn_idx is less than the position of the 1st element or greater than the position of the last element
                    if value_elem_span[0] <= attn_idx <= value_elem_span[1]:
                        if slot.get('is_boolean_neg', False):
                            if slot['mentioned'][elem_idx] is False:
                                slot['mentioned'][elem_idx] = None
                            else:
                                slot['mentioned'][elem_idx] = True
                        else:
                            slot['mentioned'][elem_idx] = True

                        if not slot['confidence'][elem_idx]:
                            slot['confidence'][elem_idx] = confidence
                        attn_weight_matched = True

                        break

            if 'value_span' not in slot or slot['is_boolean']:
                # For Boolean slots and slots without a value, match the slot's name
                if slot['name_span'][0] <= attn_idx <= slot['name_span'][1]:
                    if slot.get('is_boolean_neg', False):
                        if slot['mentioned'][0] is False:
                            slot['mentioned'][0] = None
                        else:
                            slot['mentioned'][0] = True
                    else:
                        slot['mentioned'][0] = True

                    if not slot['confidence'][0]:
                        slot['confidence'][0] = confidence
                    attn_weight_matched = True

            if attn_weight_matched:
                break


def remove_slot_mentions(slot_mention_batch, attn_idxs):
    for batch_idx, attn_idx in zip(attn_idxs[0].tolist(), attn_idxs[-1].tolist()):
        for slot in slot_mention_batch[batch_idx]:
            if not any(slot['mentioned']):
                continue

            attn_weight_matched = False

            if 'value_span' in slot and not slot['is_boolean']:
                # Check if attn_idx is between the position of the 1st element and the position of the last element
                if slot['value_span'][0][0] <= attn_idx <= slot['value_span'][-1][1]:
                    for elem_idx, value_elem_span in enumerate(slot['value_span']):
                        if value_elem_span[0] <= attn_idx <= value_elem_span[1]:
                            if not slot['confidence'][elem_idx]:
                                slot['mentioned'][elem_idx] = False
                            attn_weight_matched = True
                            break
            else:
                # For Boolean slots and slots without a value, match the slot's name
                if slot['name_span'][0] <= attn_idx <= slot['name_span'][1]:
                    if not slot['confidence'][0]:
                        slot['mentioned'][0] = False
                    attn_weight_matched = True

            if attn_weight_matched:
                break


def evaluate_slot_mentions(slot_mentions_batch):
    slot_errors_batch = []

    for slot_mentions_beam in slot_mentions_batch:
        slot_errors_beam = []

        for slot_mentions in slot_mentions_beam:
            # If any of the slot's values were not mentioned, consider the slot mention erroneous
            slot_errors = [slot['name'] for slot in slot_mentions if not all(slot['mentioned'])]
            slot_errors_beam.append(slot_errors)

        slot_errors_batch.append(slot_errors_beam)

    return slot_errors_batch


def aggregate_across_layers(attn_weights, mode='max'):
    """Aggregates weights across all attention layers."""
    if mode == 'max':
        layer_sums, _ = attn_weights.max(dim=1, keepdim=True)
    elif mode == 'sum':
        layer_sums = attn_weights.sum(dim=1, keepdim=True)
    elif mode == 'avg':
        layer_sums = attn_weights.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f'Aggregation mode "{mode}" unrecognized')

    return layer_sums


def aggregate_across_heads(attn_weights, mode='max'):
    """Aggregates weights across all attention heads."""
    if mode == 'max':
        head_maxs, _ = attn_weights.max(dim=2, keepdim=True)
    elif mode == 'sum':
        head_maxs = attn_weights.sum(dim=2, keepdim=True)
    elif mode == 'avg':
        head_maxs = attn_weights.mean(dim=2, keepdim=True)
    else:
        raise ValueError(f'Aggregation mode "{mode}" unrecognized')

    return head_maxs


def binarize_weights(attn_weights, threshold=0.0, keep_max_only=False):
    # attn_weights_bin = attn_weights.detach().clone()
    attn_weights_bin = attn_weights

    if keep_max_only:
        max_weights, _ = attn_weights_bin.max(dim=-1, keepdim=True)

        attn_weights_bin[torch.nonzero(attn_weights_bin < threshold, as_tuple=True)] = 0
        attn_weights_bin = (attn_weights_bin == max_weights).int()
    else:
        attn_weights_bin[torch.nonzero(attn_weights_bin < threshold, as_tuple=True)] = 0
        attn_weights_bin[torch.nonzero(attn_weights_bin >= threshold, as_tuple=True)] = 1

    return attn_weights_bin


def rearrange_slot_mentions_for_next_time_step(slot_spans, beam_idxs):
    return [copy.deepcopy(slot_spans[beam_idx]) for beam_idx in beam_idxs]
