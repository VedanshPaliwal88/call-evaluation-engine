"""Text normalization utilities for profanity matching and verification parsing."""
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
    """Normalize text for profanity pattern matching.

    Applies a multi-step pipeline to collapse evasions (leet-speak, starred
    characters, character repetition) so that a single lexicon pattern catches
    all common variants.

    Args:
        text: Raw utterance text from a transcript turn.

    Returns:
        Lowercase, whitespace-collapsed string ready for regex matching.
    """
    # Step 1: lowercase + translate leet substitutions (e.g. '$' → 's', '0' → 'o')
    lowered = text.lower().translate(LEET_MAP)

    # Step 2: expand starred profanity (e.g. "f**k" → "fuck") before stripping symbols
    for pattern, replacement in STARRED_PROFANITY:
        lowered = pattern.sub(replacement, lowered)

    # Step 3: collapse character-repetition evasions (e.g. "fuuuck" → "fuck")
    lowered = REPEATED_CHARS.sub(r"\1", lowered)

    # Step 4: strip punctuation and non-alphabetic characters, keeping apostrophes
    lowered = NON_ALPHA.sub(" ", lowered)

    # Step 5: collapse internal whitespace
    lowered = WHITESPACE.sub(" ", lowered).strip()

    # Step 6: expand informal abbreviations and split compound words that would
    # otherwise miss individual pattern hits (e.g. "bullshit" → "bull shit")
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
    # Wrap in spaces so replacements only match whole tokens, not substrings
    normalized = f" {lowered} "
    for source, target in replacements.items():
        normalized = normalized.replace(f" {source} ", f" {target} ")
    return WHITESPACE.sub(" ", normalized).strip()


def normalize_for_verification(text: str) -> str:
    """Normalize text for identity verification pattern matching.

    Lighter than `normalize_for_matching`; preserves digits, slashes, and
    hyphens so that date and SSN patterns remain matchable.

    Args:
        text: Raw utterance text from a transcript turn.

    Returns:
        Lowercase, punctuation-stripped string suitable for verification regexes.
    """
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s/\-']", " ", lowered)
    return WHITESPACE.sub(" ", lowered).strip()
