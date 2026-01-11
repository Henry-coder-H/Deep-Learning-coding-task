from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from pipeline.utils import CRNN_ALPHABET


class CTCLabelConverter:
    """Encode/Decode between text-label and CTC tensor."""

    def __init__(self, alphabet: Iterable[str]):
        self.alphabet = list(alphabet)
        self.blank = len(self.alphabet)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.alphabet)}

    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(t) for t in texts]
        total_len = sum(lengths)
        targets = torch.full((total_len,), self.blank, dtype=torch.long)
        idx = 0
        for text in texts:
            for ch in text:
                targets[idx] = self.char_to_idx.get(ch, self.blank - 1)
                idx += 1
        return targets, torch.tensor(lengths, dtype=torch.long)

    def decode(self, preds: torch.Tensor, pred_lens: torch.Tensor) -> List[str]:
        """Greedy decode. preds: [T, B, C], pred_lens: [B]."""
        max_idx = preds.argmax(2)  # [T, B]
        max_idx = max_idx.permute(1, 0)  # [B, T]
        texts = []
        for idxs, plen in zip(max_idx, pred_lens):
            seq = idxs[: plen].tolist()
            # Collapse repeats and blanks
            prev = self.blank
            chars = []
            for s in seq:
                if s != prev and s != self.blank and s < len(self.alphabet):
                    chars.append(self.alphabet[s])
                prev = s
            texts.append("".join(chars))
        return texts


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

