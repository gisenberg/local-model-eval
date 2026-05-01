"""Utility class for common string manipulations."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A collection of static‑style string processing helpers.

    The methods are deliberately kept simple and side‑effect free so they can be
    easily unit‑tested.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are delimited by whitespace (any amount of spaces, tabs or newlines).

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            The words of ``s`` in reverse order, joined by a single space.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowel characters (a, e, i, o, u) in the string,
        case‑insensitively.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        int
            Number of vowel characters.
        """
        vowels = set("aeiou")
        return sum(1 for ch in s.lower() if ch in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Determine whether ``s`` is a palindrome when ignoring case,
        whitespace and punctuation.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if the cleaned string reads the same forwards and backwards.
        """
        # Keep only alphanumeric characters and lower‑case them.
        cleaned = re.sub(r"[^A-Za-z0-9]", "", s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Encode ``s`` with a Caesar cipher that shifts alphabetic characters.

        Only letters ``a‑z`` and ``A‑Z`` are shifted; all other characters are left
        untouched.  ``shift`` may be positive or negative and is taken modulo 26.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift each letter.

        Returns
        -------
        str
            The cipher‑text.
        """
        shift %= 26  # Normalise to range [0, 25]
        result = []

        for ch in s:
            if ch.islower():
                base = ord("a")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            elif ch.isupper():
                base = ord("A")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in ``s`` (case‑insensitive).  If several
        words share the highest frequency, the one that appears first in the
        original text is returned.  Returns ``None`` when the string contains no
        words.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word, or ``None`` if there are no words.
        """
        # Split on whitespace; treat punctuation attached to a word as part of it.
        words = s.split()
        if not words:
            return None

        # Normalise to lower‑case for case‑insensitivity.
        lowered = [w.lower() for w in words]

        # Counter gives frequencies; we need the first occurrence in case of ties.
        counts = Counter(lowered)
        max_freq = max(counts.values())

        # Iterate original order to respect first‑appearance tie‑breaking.
        for original_word in words:
            if counts[original_word.lower()] == max_freq:
                return original_word
        # Unreachable, but keep the type‑checker happy.
        return None


# ----------------------------------------------------------------------
# Pytest test suite
# ----------------------------------------------------------------------
import pytest


@pytest.fixture
def processor():
    """Provide a fresh instance of StringProcessor for each test."""
    return StringProcessor()


def test_reverse_words():
    assert StringProcessor.reverse_words("  hello   world  ") == "world hello"
    assert StringProcessor.reverse_words("single") == "single"
    assert StringProcessor.reverse_words("") == ""


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3  # e, o, o
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False
    assert StringProcessor.is_palindrome("") is True  # empty string is a palindrome


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("xyz", 3) == "abc"
    assert StringProcessor.caesar_cipher("ABC", -1) == "ZAB"
    assert StringProcessor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    # shift larger than alphabet size
    assert StringProcessor.caesar_cipher("hello", 27) == "mjqqt"


def test_most_common_word():
    txt = "The quick brown fox jumps over the lazy dog the fox"
    # "the" appears 3 times (case‑insensitive), "fox" appears 2 times.
    assert StringProcessor.most_common_word(txt) == "the"

    # Tie‑breaking: "cat" and "dog" both appear twice, but "cat" comes first.
    txt2 = "cat dog cat dog"
    assert StringProcessor.most_common_word(txt2) == "cat"

    # No words → None
    assert StringProcessor.most_common_word("   ") is None