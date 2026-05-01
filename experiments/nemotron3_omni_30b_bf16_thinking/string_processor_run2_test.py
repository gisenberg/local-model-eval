"""StringProcessor – a small utility class for common string manipulations."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional


class StringProcessor:
    """
    A collection of static‑style string utilities.

    The class does not maintain any state; all methods are static and can be
    called directly on the class (or on an instance).
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are defined as substrings separated by whitespace.
        The original whitespace layout is not preserved – the result is a
        single space between words.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            The words of *s* in reverse order.
        """
        words: List[str] = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowel characters (a, e, i, o, u) in *s*,
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
        Determine whether *s* is a palindrome, ignoring case,
        spaces and any punctuation characters.

        Only alphabetic characters are considered.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if the filtered string reads the same forward and backward.
        """
        # Keep only alphabetic characters and lower‑case them.
        filtered = [ch.lower() for ch in s if ch.isalpha()]
        return filtered == list(reversed(filtered))

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* shifting alphabetic characters by *shift*.
        Non‑alphabetic characters are left untouched.

        The shift may be positive or negative and wraps around the alphabet.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift each letter.

        Returns
        -------
        str
            The ciphered string.
        """
        result: List[str] = []
        for ch in s:
            if ch.isalpha():
                base = ord('a') if ch.islower() else ord('A')
                # Normalise to 0‑25, apply shift, wrap with modulo 26, then restore.
                offset = (ord(ch) - base + shift) % 26
                result.append(chr(base + offset))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in *s* (case‑insensitive).
        If several words share the highest frequency, the one that appears
        first in the original string is returned.

        Words are sequences of alphanumeric characters separated by
        whitespace; punctuation attached to a word is stripped.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word, or ``None`` if *s* contains no words.
        """
        # Extract words – this removes punctuation but keeps internal digits.
        words = re.findall(r"\b\w+\b", s.lower())
        if not words:
            return None

        # Counter gives frequencies; we need the first word with max count.
        counts = Counter(words)
        max_count = max(counts.values())
        for w in words:               # preserves original order
            if counts[w] == max_count:
                return w
        return None  # unreachable, but keeps mypy happy


# --------------------------------------------------------------------------- #
#                               Pytest tests                                 #
# --------------------------------------------------------------------------- #

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  keep   the   spaces  ") == "spaces the keep"
    assert StringProcessor.reverse_words("") == ""

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3   # e, o, o
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10
    assert StringProcessor.count_vowels("") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False
    assert StringProcessor.is_palindrome("") is True   # empty string is trivially a palindrome

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("abc XYZ", -1) == "zab WXY"
    assert StringProcessor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    # wrap around
    assert StringProcessor.caesar_cipher("xyz", 2) == "zab"

def test_most_common_word():
    txt = "The quick brown fox jumps over the lazy dog the fox was quick"
    # 'the' appears 3 times, 'quick' 2, 'fox' 2 → 'the' is the answer
    assert StringProcessor.most_common_word(txt) == "the"
    # tie – first occurrence wins
    txt2 = "cat dog cat dog"
    assert StringProcessor.most_common_word(txt2) == "cat"
    assert StringProcessor.most_common_word("") is None