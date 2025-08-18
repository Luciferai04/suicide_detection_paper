from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i've": "i have",
    "we're": "we are",
    "they're": "they are",
}


def _expand_contractions(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        w = match.group(0).lower()
        return CONTRACTIONS.get(w, w)

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b", re.IGNORECASE)
    return pattern.sub(repl, text)


def _squeeze_elongations(text: str, max_repeats: int = 2) -> str:
    # e.g., 'soooo' -> 'soo'
    return re.sub(r"(\w)\1{2,}", lambda m: m.group(1) * max_repeats, text)


@dataclass
class CleanConfig:
    lowercase: bool = True
    expand_contractions: bool = True
    squeeze_elongations: bool = True
    preserve_emphasis: bool = True  # Do not remove !!! or ???
    remove_extra_spaces: bool = True


def build_cleaner(cfg: CleanConfig) -> Callable[[str], str]:
    def cleaner(text: str) -> str:
        s = text if isinstance(text, str) else str(text)
        if cfg.expand_contractions:
            s = _expand_contractions(s)
        if cfg.squeeze_elongations:
            s = _squeeze_elongations(s)
        # Normalize whitespace
        if cfg.remove_extra_spaces:
            s = re.sub(r"\s+", " ", s).strip()
        if cfg.lowercase:
            s = s.lower()
        return s

    return cleaner
