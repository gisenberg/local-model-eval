"""Utility class for common string processing tasks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A small collection of static‑style string processing helpers.

    The methods are deliberately pure (they do not modify the input) and
    include full type hints and documentation strings.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are delimited by any whitespace (space, tab, newline …).  The
        resulting string uses a single space between words.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            The words of ``s`` in reverse order.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowel characters (a, e, i, o, u) in ``s``.

        The check is case‑insensitive.

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
        Determine whether ``s`` is a palindrome when case, spaces,
        and punctuation are ignored.

        Only alphanumeric characters are considered.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if the filtered string reads the same forwards and
            backwards, ``False`` otherwise.
        """
        filtered = [ch.lower() for ch in s if ch.isalnum()]
        return filtered == list(reversed(filtered))

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to ``s`` shifting alphabetic characters by
        ``shift`` positions.  Alphabetic characters are wrapped within the
        alphabet; non‑alphabetic characters are left untouched.

        The function supports positive and negative shifts.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift each letter.

        Returns
        -------
        str
            The cipher‑transformed string.
        """
        result = []

        for ch in s:
            if 'a' <= ch <= 'z':
                base = ord('a')
                new_char = chr((ord(ch) - base + shift) % 26 + base)
                result.append(new_char)
            elif 'A' <= ch <= 'Z':
                base = ord('A')
                new_char = chr((ord(ch) - base + shift) % 26 + base)
                result.append(new_char)
            else:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in ``s`` (case‑insensitive).  If several
        words share the highest frequency, the one that appears first in the
        original string is returned.

        Parameters
        ----------
        s: str
            Input string.  Words are delimited by whitespace.

        Returns
        -------
        Optional[str]
            The most common word in its original capitalisation, or ``None``
            if ``s`` is empty or contains no words.
        """
        if not s.strip():
            return None

        # Split on whitespace, keep original word for tie‑breaking.
        words = s.split()
        lowered = [w.lower() for w in words]

        # Count frequencies.
        counts = Counter(lowered)
        if not counts:
            return None

        max_freq = max(counts.values())

        # Preserve the first occurrence of each word (original case).
        first_occurrence: dict[str, str] = {}
        for original, low in zip(words, lowered):
            if low not in first_occurrence:
                first_occurrence[low] = original

        # Find the word(s) with max frequency and pick the earliest.
        candidates = [word for word, cnt in counts.items() if cnt == max_freq]
        # Sort candidates by their first appearance index.
        candidates.sort(key=lambda w: lowered.index(w))

        return first_occurrence[candidates[0]]

import pytest



def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  many   spaces  here  ") == "here spaces many"


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3   # e, o, o
    assert StringProcessor.count_vowels("AEIOU") == 5
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
    assert StringProcessor.is_palindrome("No lemon, no melon")
    assert not StringProcessor.is_palindrome("Hello, World!")


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd", -1) == "abc"
    # non‑letters are unchanged
    assert StringProcessor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"


def test_most_common_word():
    txt = "The the cat cat dog"
    # "the" and "cat" each appear twice; "the" appears first.
    assert StringProcessor.most_common_word(txt) == "The"
    # empty string -> None
    assert StringProcessor.most_common_word("") is None
    # single word
    assert StringProcessor.most_common_word("Solo") == "Solo"