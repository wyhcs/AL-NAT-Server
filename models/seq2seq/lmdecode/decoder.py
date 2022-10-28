import heapq
import math
import os
import sentencepiece as spm
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import kenlm

from models.seq2seq.lmdecode.alphabet import BPE_CHAR, Alphabet
from models.seq2seq.lmdecode.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    MIN_TOKEN_CLIP_P,
)
from models.seq2seq.lmdecode.language_model import AbstractLanguageModel, LanguageModel

# type hints
Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]
Beam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float]
LMBeam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float, float]
LMState = Optional[Union[kenlm.State, List[kenlm.State]]]
OutputBeam = Tuple[str, LMState, List[WordFrames], float, float]

# constants
NULL_FRAMES: Frames = (-1, -1)
EMPTY_START_BEAM: Beam = ("", "", "", None, [], NULL_FRAMES, 0.0)

def _sort_and_trim_beams(beams: List[LMBeam], beam_width: int) -> List[LMBeam]:
    return heapq.nlargest(beam_width, beams, key=lambda x: x[-1])


def _sum_log_scores(s1: float, s2: float) -> float:
    if s1 >= s2:
        log_sum = s1 + math.log(1 + math.exp(s2 - s1))
    else:
        log_sum = s2 + math.log(1 + math.exp(s1 - s2))
    return log_sum


def _log_softmax(
    x: np.ndarray,  # type: ignore [type-arg]
    axis: Optional[int] = None,
) -> np.ndarray:  # type: ignore [type-arg]
    """Logarithm of softmax function, following implementation of scipy.special."""
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0  # pylint: disable=R0204
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray = np.log(s)  # type: ignore [type-arg]
    out = tmp - out
    return out

def _merge_tokens(token_1: str, token_2: str) -> str:
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return text

def _merge_beams(beams: List[Beam]) -> List[Beam]:
    beam_dict = {}
    for text, next_word, word_part, last_char, text_frames, part_frames, logit_score in beams:
        new_text = _merge_tokens(text, next_word)
        hash_idx = (new_text, word_part, last_char)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
            )
        else:
            beam_dict[hash_idx] = (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                _sum_log_scores(beam_dict[hash_idx][-1], logit_score),
            )
    return list(beam_dict.values())


def _prune_history(beams: List[LMBeam], lm_order: int) -> List[Beam]:
    min_n_history = max(1, lm_order - 1)
    seen_hashes = set()
    filtered_beams = []
    for (text, next_word, word_part, last_char, text_frames, part_frames, logit_score, _) in beams:
        hash_idx = (tuple(text.split()[-min_n_history:]), word_part, last_char)
        if hash_idx not in seen_hashes:
            filtered_beams.append(
                (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                )
            )
            seen_hashes.add(hash_idx)
    return filtered_beams


def Wp2idx(sp, token2idx, text):
    wps = sp.EncodeAsPieces(text)
    token_ids = []
    for wp in wps:
        if wp in token2idx.keys():
            token_ids.append(token2idx[wp])
        else:
            token_ids.append(token2idx['<unk>'])
    return token_ids


def Char2idx(token2idx, text):
    token_ids = []
    for wp in list(text):
        if wp in token2idx.keys():
            token_ids.append(token2idx[wp])
        else:
            token_ids.append(token2idx['<unk>'])
    return token_ids


class BeamSearchDecoderCTC:
    model_container: Dict[bytes, Optional[AbstractLanguageModel]] = {}
    def __init__(
        self,
        alphabet: Alphabet,
        language_model: Optional[AbstractLanguageModel] = None,
        bpe_path = None,
    ) -> None:
        self._alphabet = alphabet
        self._idx2vocab = {n: c for n, c in enumerate(self._alphabet.labels)}
        self.vocab2idx = {c: n for n, c in enumerate(self._alphabet.labels)}

        self._is_bpe = alphabet.is_bpe
        self.bpe_path = bpe_path
        self._model_key = os.urandom(16)
        BeamSearchDecoderCTC.model_container[self._model_key] = language_model

    def _get_lm_beams(
        self,
        beams: List[Beam],
        cached_lm_scores: Dict[str, Tuple[float, LMState]],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        language_model = BeamSearchDecoderCTC.model_container[self._model_key]
        if language_model is None:
            new_beams = []
            for text, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
                new_text = _merge_tokens(text, next_word)
                new_beams.append(
                    (
                        new_text,
                        "",
                        word_part,
                        last_char,
                        frame_list,
                        frames,
                        logit_score,
                    )
                )
            return new_beams

        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score in beams:
            new_text = _merge_tokens(text, next_word)
            if new_text not in cached_lm_scores:
                prev_raw_lm_score, start_state = cached_lm_scores[text]
                score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                raw_lm_score = prev_raw_lm_score + score
                cached_lm_scores[new_text] = (raw_lm_score, end_state)

            lm_score, _ = cached_lm_scores[new_text]

            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    cached_partial_token_scores[word_part] = language_model.score_partial_token(word_part)
                lm_score += cached_partial_token_scores[word_part]

            new_beams.append(
                (
                    new_text,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                )
            )

        return new_beams

    def _decode_logits(
        self,
        logits: np.ndarray,
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        lm_start_state: LMState = None,
    ) -> List[OutputBeam]:
        if math.isclose(int(logits.sum(axis=1).mean()), 1):
            # input looks like probabilities, so take log
            logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
        else:
            # convert logits into log probs
            logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)

        language_model = BeamSearchDecoderCTC.model_container[self._model_key]
        if lm_start_state is None and language_model is not None:
            cached_lm_scores: Dict[str, Tuple[float, LMState]] = {
                "": (0.0, language_model.get_start_state())
            }
        else:
            cached_lm_scores = {"": (0.0, lm_start_state)}
        
        cached_p_lm_scores: Dict[str, float] = {}
        beams = [EMPTY_START_BEAM]
        force_next_break = False
        for frame_idx, logit_col in enumerate(logits):
            max_idx = logit_col.argmax()
            idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
            new_beams: List[Beam] = []
            for idx_char in idx_list:
                p_char = logit_col[idx_char]
                char = self._idx2vocab[idx_char]
                for (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                ) in beams:
                    if char == "" or last_char == char:
                        new_part_frames = (
                            part_frames if char == "" else (part_frames[0], frame_idx + 1)
                        )
                        new_beams.append(
                            (
                                text,
                                next_word,
                                word_part,
                                char,
                                text_frames,
                                new_part_frames,
                                logit_score + p_char,
                            )
                        )
                    elif self._is_bpe and (char[:1] == BPE_CHAR or force_next_break):
                        force_next_break = False
                        clean_char = char
                        if char[:1] == BPE_CHAR:
                            clean_char = clean_char[1:]
                        if char[-1:] == BPE_CHAR:
                            clean_char = clean_char[:-1]
                            force_next_break = True
                        new_frame_list = (
                            text_frames
                            if word_part == ""
                            else text_frames + [(part_frames[0], frame_idx)]
                        )
                        new_beams.append(
                            (
                                text,
                                word_part,
                                clean_char,
                                char,
                                new_frame_list,
                                (-1, -1),
                                logit_score + p_char,
                            )
                        )
                    elif not self._is_bpe and char == " ":
                        new_frame_list = (
                            text_frames if word_part == "" else text_frames + [part_frames]
                        )
                        new_beams.append(
                            (
                                text,
                                word_part,
                                "",
                                char,
                                new_frame_list,
                                NULL_FRAMES,
                                logit_score + p_char,
                            )
                        )
                    else:
                        new_part_frames = (
                            (frame_idx, frame_idx + 1)
                            if part_frames[0] < 0
                            else (part_frames[0], frame_idx + 1)
                        )
                        new_beams.append(
                            (
                                text,
                                next_word,
                                word_part + char,
                                char,
                                text_frames,
                                new_part_frames,
                                logit_score + p_char,
                            )
                        )
            new_beams = _merge_beams(new_beams)
            scored_beams = self._get_lm_beams(
                new_beams,
                cached_lm_scores,
                cached_p_lm_scores,
            )
            max_score = max([b[-1] for b in scored_beams])
            scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
            trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
            if prune_history:
                lm_order = 1 if language_model is None else language_model.order
                beams = _prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [b[:-1] for b in trimmed_beams]

        new_beams = []
        for text, _, word_part, _, frame_list, frames, logit_score in beams:
            new_token_times = frame_list if word_part == "" else frame_list + [frames]
            new_beams.append((text, word_part, "", None, new_token_times, (-1, -1), logit_score))
        new_beams = _merge_beams(new_beams)
        scored_beams = self._get_lm_beams(
            new_beams,
            cached_lm_scores,
            cached_p_lm_scores,
            is_eos=True,
        )

        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        output_beams = [
            (
                " ".join(text.split()),
                cached_lm_scores[text][-1] if text in cached_lm_scores else None,
                list(zip(text.split(), text_frames)),
                logit_score,
                lm_score,
            )
            for text, _, _, _, text_frames, _, logit_score, lm_score in trimmed_beams
        ]
        return output_beams

    def decode(
        self,
        logits: np.ndarray,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        lm_start_state: LMState = None,
        is_train=True,
    ) -> str:
        decoded_beams = self._decode_logits(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=True,
            lm_start_state=lm_start_state,
        )

        if is_train:
            if self._is_bpe:
                sp = spm.SentencePieceProcessor()
                sp.Load(self.bpe_path)
                return Wp2idx(sp, self.vocab2idx, decoded_beams[0][0])
            else:
                return Char2idx(self.vocab2idx, decoded_beams[0][0])
        else:
            return decoded_beams[0][0] 

def build_ctcdecoder(
    labels: List[str],
    kenlm_model: Optional[kenlm.Model] = None,
    unigrams: Optional[Iterable[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
    lm_score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
    ctc_token_idx: Optional[int] = None,
    is_bpe: bool = False,
    bpe_path: str = None,
    ):
    if is_bpe:
        alphabet = Alphabet.build_bpe_alphabet(labels, ctc_token_idx=ctc_token_idx)
    else:
        alphabet = Alphabet.build_alphabet(labels, ctc_token_idx=ctc_token_idx)
    if kenlm_model is not None:
        language_model: Optional[AbstractLanguageModel] = LanguageModel(
            kenlm_model,
            unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=lm_score_boundary,
        )
    else:
        language_model = None
    return BeamSearchDecoderCTC(alphabet, language_model, bpe_path)
