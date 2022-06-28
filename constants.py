# ---- Data processing ----

class SlotNameConversionMode(object):
    SPECIAL_TOKENS = 'special_tokens'
    VERBALIZE = 'verbalize'


# ---- Metrics ----

class BertScoreModelCheckpoint(object):
    DEBERTA_XLARGE_MNLI = 'microsoft/deberta-xlarge-mnli'
    DEBERTA_LARGE_MNLI = 'microsoft/deberta-large-mnli'


class BleurtModelPath(object):
    BLEURT_20 = 'eval/BLEURT/models/BLEURT-20'
    BLEURT_20_D12 = 'eval/BLEURT/models/BLEURT-20-D12'
    BLEURT_20_D6 = 'eval/BLEURT/models/BLEURT-20-D6'
    BLEURT_20_D3 = 'eval/BLEURT/models/BLEURT-20-D3'


class PerplexityModelId(object):
    BART_BASE = 'facebook/bart-base'
    BART_LARGE = 'facebook/bart-large'
    GPT2 = 'gpt2'
    GPT2_MEDIUM = 'gpt2-medium'
    GPT2_LARGE = 'gpt2-large'
    XLNET_BASE = 'xlnet-base-cased'
    XLNET_LARGE = 'xlnet-large-cased'
