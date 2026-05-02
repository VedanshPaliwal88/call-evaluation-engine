from __future__ import annotations

import re


REPEATED_CHARS = re.compile(r"(.)\1{2,}")
STARRED_PROFANITY = [
    (re.compile(r"\bf[\*\s_]+ck\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf(?:[\*\s_]){2,}\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bbull[\*\s_]*shit\b", re.IGNORECASE), "bullshit"),
]
NON_ALPHA = re.compile(r"[^a-z0-9\s']")
WHITESPACE = re.compile(r"\s+")

LEET_MAP = str.maketrans(
    {
        "@": "a",
        "$": "s",
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "!": "i",
    }
)


def normalize_for_matching(text: str) -> str:
    lowered = text.lower().translate(LEET_MAP)
    for pattern, replacement in STARRED_PROFANITY:
        lowered = pattern.sub(replacement, lowered)
    lowered = REPEATED_CHARS.sub(r"\1", lowered)
    lowered = NON_ALPHA.sub(" ", lowered)
    lowered = WHITESPACE.sub(" ", lowered).strip()
    replacements = {
        "u r": "you are",
        "ur ": "your ",
        "fckn": "fucking",
        "fck": "fuck",
        "fkn": "fucking",
        "wtf": "what the fuck",
        "bullshit": "bull shit",
        "dum": "dumb",
        "stupd": "stupid",
        "as hle": "asshole",
    }
    normalized = f" {lowered} "
    for source, target in replacements.items():
        normalized = normalized.replace(f" {source} ", f" {target} ")
    return WHITESPACE.sub(" ", normalized).strip()
