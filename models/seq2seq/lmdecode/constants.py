import math

DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 1.5
DEFAULT_UNK_LOGP_OFFSET = -10.0
DEFAULT_BEAM_WIDTH = 10
DEFAULT_PRUNE_LOGP = -10.0
DEFAULT_MIN_TOKEN_LOGP = -5.0
DEFAULT_SCORE_LM_BOUNDARY = True
AVG_TOKEN_LEN = 6
MIN_TOKEN_CLIP_P = 1e-15
LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)
