"""Utility class for common string processing tasks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A collection of static‑style string processing helpers.

    The methods are deliberately pure (no side‑effects) and include full
    type hints and documentation to make the API clear and IDE‑friendly.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in *s*.

        Words are delimited by any whitespace.  The returned string contains
        a single space between words, regardless of the original spacing.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            The words of *s* in reverse order.
        """
        words = s.split()
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
        vowels = set("aeiouAEIOU")
        return sum(1 for ch in s if ch in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Determine whether *s* is a palindrome, ignoring case,
        spaces and all punctuation/non‑alphanumeric characters.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if the cleaned string reads the same forward and backward.
        """
        cleaned = [ch.lower() for ch in s if ch.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* using the given *shift*.

        Only alphabetic characters (A‑Z, a‑z) are shifted; all other
        characters are left unchanged.  The shift may be negative.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift each letter.  Positive shifts move
            forward in the alphabet, negative shifts move backward.

        Returns
        -------
        str
            The cipher‑transformed string.
        """
        result = []

        for ch in s:
            if "a" <= ch <= "z":
                base = ord("a")
                new_char = chr((ord(ch) - base + shift) % 26 + base)
                result.append(new_char)
            elif "A" <= ch <= "Z":
                base = ord("A")
                new_char = chr((ord(ch) - base + shift) % 26 + base)
                result.append(new_char)
            else:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in *s* (case‑insensitive).

        Words are sequences of alphanumeric characters separated by
        non‑word characters.  If several words share the highest frequency,
        the word that appears first in the original string is returned.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word, or ``None`` if *s* contains no words.
        """
        # Extract words in their original order (lower‑cased for case‑insensitivity)
        words = re.findall(r"\b\w+\b", s.lower())
        if not words:
            return None

        counts = Counter(words)
        max_count = max(counts.values())

        # Return the first word (in original order) that has the max count
        for w in words:
            if counts[w] == max_count:
                return w
        # Unreachable, but kept for type‑checkers
        return None

"""Pytest test suite for :class:`StringProcessor`."""

import pytest



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
    assert StringProcessor.is_palindrome("") is True  # empty string is trivially a palindrome


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    # negative shift
    assert StringProcessor.caesar_cipher("Khoor, Zruog!", -3) == "Hello, World!"
    # wrap‑around
    assert StringProcessor.caesar_cipher("xyz XYZ", 2) == "zab ZAB"
    # non‑letters unchanged
    assert StringProcessor.caesar_cipher("Hello 123!", 5) == "Mjqqt 123!"


def test_most_common_word():
    txt = "The cat and the dog. The cat was happy."
    # Words (case‑insensitive): the, cat, and, the, dog, the, cat, was, happy
    # Frequencies: the=3, cat=2, others=1 → "the" is the answer
    assert StringProcessor.most_common_word(txt) == "the"

    # Tie‑breaking: "apple" and "banana" appear twice, but "apple" occurs first.
    tie_txt = "Apple banana apple Banana"
    assert StringProcessor.most_common_word(tie_txt) == "apple"

    # Empty string → None
    assert StringProcessor.most_common_word("") is None