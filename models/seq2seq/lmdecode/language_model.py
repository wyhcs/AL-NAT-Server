import abc
import re
import math
from typing import Iterable, List, Optional, Tuple, cast

import numpy as np
from pygtrie import CharTrie
import kenlm

AVG_TOKEN_LEN = 6
LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)


from models.seq2seq.lmdecode.constants import (
    AVG_TOKEN_LEN,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    LOG_BASE_CHANGE_FACTOR,
)


def _get_empty_lm_state() -> kenlm.State:
    try:
        kenlm_state = kenlm.State()
    except ImportError:
        raise ValueError("To use a language model, you need to install kenlm.")
    return kenlm_state


class AbstractLanguageModel(abc.ABC):
    @property
    @abc.abstractmethod
    def order(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_start_state(self) -> List[kenlm.State]:
        raise NotImplementedError()

    @abc.abstractmethod
    def score_partial_token(self, partial_token: str) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def score(
        self, prev_state: kenlm.State, word: str, is_last_word: bool = False
    ) -> Tuple[float, kenlm.State]:
        raise NotImplementedError()


class LanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        kenlm_model: kenlm.Model,
        unigrams: Optional[Iterable[str]] = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
        score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
    ) -> None:
        self._kenlm_model = kenlm_model
        if unigrams is None:
            unigram_set = set()
            char_trie = None
        else:
            unigram_set = set([t for t in set(unigrams) if t in self._kenlm_model])
            char_trie = CharTrie.fromkeys(unigram_set)
        self._unigram_set = unigram_set
        self._char_trie = char_trie
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @property
    def order(self) -> int:
        return cast(int, (self._kenlm_model.order))

    def get_start_state(self) -> kenlm.State:
        start_state = _get_empty_lm_state()
        if self.score_boundary:
            self._kenlm_model.BeginSentenceWrite(start_state)
        else:
            self._kenlm_model.NullContextWrite(start_state)
        return start_state

    def _get_raw_end_score(self, start_state: kenlm.State) -> float:
        if self.score_boundary:
            end_state = _get_empty_lm_state()
            score: float = self._kenlm_model.BaseScore(start_state, "</s>", end_state)
        else:
            score = 0.0
        return score

    def score_partial_token(self, partial_token: str) -> float:
        if self._char_trie is None:
            return 0.0
        unk_score = self.unk_score_offset * int(self._char_trie.has_node(partial_token) == 0)
        if len(partial_token) > AVG_TOKEN_LEN:
            unk_score = unk_score * len(partial_token) / AVG_TOKEN_LEN
        return unk_score

    def score(
        self, prev_state: kenlm.State, word: str, is_last_word: bool = False
    ) -> Tuple[float, kenlm.State]:
        end_state = _get_empty_lm_state()
        lm_score = self._kenlm_model.BaseScore(prev_state, word, end_state)
        if (
            len(self._unigram_set) > 0
            and word not in self._unigram_set
            or word not in self._kenlm_model
        ):
            lm_score += self.unk_score_offset
        if is_last_word:
            lm_score = lm_score + self._get_raw_end_score(end_state)
        lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
        return lm_score, end_state
