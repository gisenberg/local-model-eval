import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A utility class that provides several string‑processing helpers.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are defined as sequences of characters separated by whitespace.
        The output uses a single space between words.

        Example
        -------
        >>> StringProcessor.reverse_words("  hello   world  ")
        'world hello'
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
            Number of vowels found.
        """
        vowels = set("aeiou")
        return sum(1 for ch in s.lower() if ch in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Determine whether the string is a palindrome when ignoring case,
        spaces, and punctuation.

        Only alphanumeric characters are considered.

        Example
        -------
        >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
        True
        """
        cleaned = re.sub(r'[^A-Za-z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Encode *s* using a Caesar cipher that shifts alphabetic characters
        by *shift* positions.  Only letters a‑z / A‑Z are shifted; all other
        characters are left unchanged.  The shift can be negative.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift (negative values shift left).

        Returns
        -------
        str
            The encoded string.
        """
        result = []
        for ch in s:
            if 'a' <= ch <= 'z':
                base = ord('a')
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            elif 'A' <= ch <= 'Z':
                base = ord('A')
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in *s* (case‑insensitive).  Words are
        delimited by any non‑word characters (i.e. ``\\w+`` matches a word).
        If several words share the highest frequency, the one that appears
        first in the original string is returned.  If the string contains no
        words, ``None`` is returned.

        Example
        -------
        >>> StringProcessor.most_common_word("Apple apple banana banana orange")
        'apple'
        """
        # Extract words (case‑insensitive)
        words = re.findall(r"\b\w+\b", s.lower())
        if not words:
            return None

        counts = Counter(words)
        max_freq = max(counts.values())

        # Return the first word that reaches max_freq in original order
        for w in words:
            if counts[w] == max_freq:
                return w
        return None  # unreachable, kept for type‑checkers

import pytest


def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  many   spaces  here  ") == "here spaces many"


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3   # e, o, o
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
    assert StringProcessor.is_palindrome("No lemon, no melon")
    assert not StringProcessor.is_palindrome("Hello, World!")
    assert StringProcessor.is_palindrome("12321")
    assert not StringProcessor.is_palindrome("12345")


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Hello", 1) == "Ifmmp"
    assert StringProcessor.caesar_cipher("Hello", -1) == "Gdkkn"
    assert StringProcessor.caesar_cipher("XYZ", 2) == "ZAB"
    assert StringProcessor.caesar_cipher("xyz", -2) == "vwx"
    # non‑letters are unchanged
    assert StringProcessor.caesar_cipher("Hello, World! 123", 3) == "Khoor, Zruog! 123"


def test_most_common_word():
    text = "Apple apple banana banana orange"
    assert StringProcessor.most_common_word(text) == "apple"   # tie, first wins

    # single word
    assert StringProcessor.most_common_word("singleton") == "singleton"

    # no words
    assert StringProcessor.most_common_word("   !!!  ") is None

    # mixed punctuation
    text2 = "cat, dog! cat? dog; bird"
    # counts: cat 2, dog 2, bird 1 -> first tied is "cat"
    assert StringProcessor.most_common_word(text2) == "cat"