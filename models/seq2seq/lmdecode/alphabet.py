from typing import List, Optional

BPE_CHAR = "▁"

def _get_ctc_index(label_list: List[str]) -> int:
    return len(label_list) - 1 if label_list[-1] == "" else -1

class Alphabet:
    def __init__(self, labels: List[str], is_bpe: bool) -> None:
        """Init."""
        self._labels = labels
        self._is_bpe = is_bpe

    @property
    def is_bpe(self) -> bool:
        return self._is_bpe

    @property
    def labels(self) -> List[str]:
        return self._labels[:]

    @classmethod
    def build_alphabet(
        cls, label_list: List[str], ctc_token_idx: Optional[int] = None
    ) -> "Alphabet":
        if ctc_token_idx is None:
            ctc_token_idx = _get_ctc_index(label_list)
        clean_labels = label_list[:]
        if " " not in clean_labels:
            raise ValueError("Space token ' ' missing from vocabulary.")
        if ctc_token_idx == -1:
            clean_labels.append("")
        else:
            clean_labels[ctc_token_idx] = ""
        return cls(clean_labels, False)

    @classmethod
    def build_bpe_alphabet(cls, 
        label_list: List[str],
        ctc_token_idx: Optional[int] = None,
    ) -> "Alphabet":
        if ctc_token_idx is None:
            ctc_token_idx = _get_ctc_index(label_list)
        # create copy
        formatted_label_list = label_list[:]
        return cls(formatted_label_list, True)
