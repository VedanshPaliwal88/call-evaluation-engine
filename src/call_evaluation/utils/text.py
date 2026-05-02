from __future__ import annotations

import re


REPEATED_CHARS = re.compile(r"(.)\1{2,}")
STARRED_PROFANITY = [
    (re.compile(r"\bf[\*\s_]+ck\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf\*{2,}(?=\W|$)", re.IGNORECASE), "fuck"),
    (re.compile(r"\ba[\*\s_]+hole\b", re.IGNORECASE), "asshole"),
    (re.compile(r"\bs\*{2,}(?=\W|$)", re.IGNORECASE), "shit"),
    (re.compile(r"\bs[\*\s_]+t\b", re.IGNORECASE), "shit"),
    (re.compile(r"\bb\*{3,}(?=\W|$)", re.IGNORECASE), "bitch"),
    (re.compile(r"\bb[\*\s_]+(?:ch|tch)\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bbull[\*\s_]*(?:shit|ship)\b", re.IGNORECASE), "bullshit"),
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
        "bullship": "bull shit",
        "dum": "dumb",
        "stupd": "stupid",
        "as hle": "asshole",
    }
    normalized = f" {lowered} "
    for source, target in replacements.items():
        normalized = normalized.replace(f" {source} ", f" {target} ")
    return WHITESPACE.sub(" ", normalized).strip()


def normalize_for_verification(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s/\-']", " ", lowered)
    return WHITESPACE.sub(" ", lowered).strip()
