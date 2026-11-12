"""Corpus handling: built-in starter text, loading a user corpus, and train/val/test splits."""

from __future__ import annotations

from pathlib import Path
import random

from .common import DIAGNOSTICS, SPLITS, normalized


def starter_corpus() -> list[str]:
    subjects = (
        "மாலை நேரத்தில்", "காலை நேரத்தில்", "மழை நின்ற பிறகு", "விழா முடிந்த பிறகு",
        "பள்ளி திறந்த போது", "பழைய நூலகத்தில்", "கடற்கரை அருகில்", "திண்ணை அருகில்",
        "அறையில் அமர்ந்து", "நாளை நடைபெறும் விழாவில்", "மலை அடிவாரத்தில்", "நகரின் எல்லையில்",
    )
    actors = ("மாணவன்", "மாணவி", "ஆசிரியர்", "கலைஞர்", "நண்பன்", "அண்ணன்",
              "தந்தை", "தாய்", "இளைஞன்", "கவிஞர்", "விருந்தினர்", "சிறுவன்")
    actions = ("பழைய நூலை வாசித்தார்", "கதையை எடுத்துரைத்தார்", "நல்ல செய்தியை பகிர்ந்தார்",
               "புதிய பாடலை பாடினார்", "கலை நிகழ்ச்சியை பார்த்தார்", "பணிகளை முடித்தார்",
               "கடிதத்தை எழுதினார்", "மணல் சிற்பத்தை பார்த்தார்", "விளக்கை ஏற்றினார்",
               "துணிகளை அடுக்கினார்", "எழுத்து வடிவங்களை ஆராய்ந்தார்", "மரங்களை நட்டார்")
    return [f"{s} {a} {v}." for s in subjects for a in actors for v in actions]


def load_corpus(corpus: Path | None) -> list[str]:
    """Normalized corpus lines; the built-in starter corpus when ``corpus`` is None."""
    if corpus is None:
        return starter_corpus()
    return [normalized(t) for t in corpus.read_text(encoding="utf-8-sig").splitlines() if t.strip()]


def split_texts(texts: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(normalized(t) for t in texts))
    if len(unique) < 10:
        raise ValueError("At least 10 unique corpus lines are required for train/validation/test splits.")
    random.Random(seed).shuffle(unique)
    n_val = n_test = max(1, len(unique) // 10)
    n_train = len(unique) - n_val - n_test
    return {t: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, t in enumerate(unique)}


def assign_diagnostics(assignments: dict[str, str]) -> None:
    fixed: set[str] = set()
    for text, split in zip(DIAGNOSTICS, SPLITS):
        if assignments[text] != split:
            replacement = next(t for t, s in assignments.items() if s == split and t not in fixed)
            assignments[replacement], assignments[text] = assignments[text], split
        fixed.add(text)
