"""Utility class for common string manipulations."""

from __future__ import annotations

import string
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A collection of static‑style string processing helpers.

    The methods are deliberately kept stateless – they do not store any
    instance data – which makes the class easy to use and test.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are delimited by any whitespace (space, tab, newline …) and
        the result is joined with a single space.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            Words in reverse order.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowel characters (a, e, i, o, u) in the
        given string, case‑insensitively.

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
        Determine whether *s* is a palindrome when case, spaces,
        and punctuation are ignored.

        Only alphanumeric characters are considered.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if *s* reads the same forwards and backwards after
            normalisation, otherwise ``False``.
        """
        cleaned = [ch.lower() for ch in s if ch.isalnum()]
        return cleaned == list(reversed(cleaned))

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* shifting alphabetic characters by
        ``shift`` positions.  The shift works for both lower‑ and upper‑case
        letters and wraps around the alphabet.  Non‑alphabetic characters
        are left untouched.  ``shift`` may be negative.

        Parameters
        ----------
        s: str
            Text to encode.
        shift: int
            Number of positions to shift (positive → right, negative → left).

        Returns
        -------
        str
            The encoded string.
        """
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
        Return the most common word in *s* (case‑insensitive).  If several
        words share the highest frequency, the one that appears first in
        the original text is returned.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word, or ``None`` if the string contains no
            words.
        """
        if not s.strip():
            return None

        words = s.lower().split()
        # Count frequencies while preserving first occurrence order.
        freq: Counter[str] = Counter()
        first_index: dict[str, int] = {}

        for idx, w in enumerate(words):
            freq[w] += 1
            if w not in first_index:
                first_index[w] = idx

        # Determine the highest frequency.
        if not freq:
            return None
        max_freq = max(freq.values())

        # Among the words with max frequency, pick the one with the smallest index.
        candidates = [w for w, cnt in freq.items() if cnt == max_freq]
        most_common = min(candidates, key=lambda w: first_index[w])
        return most_common

import pytest



def test_reverse_words():
    assert StringProcessor.reverse_words("the quick brown fox") == "fox brown quick the"
    assert StringProcessor.reverse_words("  hello   world  ") == "world hello"
    assert StringProcessor.reverse_words("") == ""


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3   # e, o, o
    assert StringProcessor.count_vowels("AEIOU") == 5
    assert StringProcessor.count_vowels("xyz") == 0
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False
    assert StringProcessor.is_palindrome("") is True   # empty string is trivially a palindrome


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("abc XYZ", -1) == "bcd YZA".replace("b", "a").replace("c", "b").replace("Y", "X").replace("Z", "Y")  # illustrative; better to test directly:
    # Simpler direct tests:
    assert StringProcessor.caesar_cipher("abc XYZ", 0) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert StringProcessor.caesar_cipher("Khoor, Zruog!", -3) == "Hello, World!"


def test_most_common_word():
    txt = "The quick brown fox jumps over the lazy dog the fox"
    # 'the' appears 3 times, 'fox' appears 2 times → 'the' is most common
    assert StringProcessor.most_common_word(txt) == "the"

    # Tie‑breaking: 'cat' and 'dog' both appear twice, but 'cat' appears first.
    txt2 = "cat dog cat dog"
    assert StringProcessor.most_common_word(txt2) == "cat"

    # Empty input returns None
    assert StringProcessor.most_common_word("   ") is None