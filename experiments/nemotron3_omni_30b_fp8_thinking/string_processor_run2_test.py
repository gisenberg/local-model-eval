"""StringProcessor – a small utility class for common string manipulations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A collection of static‑style string utilities.

    The class does not maintain any state; all methods are static and can be
    called directly on the class (e.g. ``StringProcessor.reverse_words("hi")``).

    Methods
    -------
    reverse_words(s):
        Reverse the order of words in *s* while preserving the original word
        characters.

    count_vowels(s):
        Count the total number of vowels (a, e, i, o, u) in *s*, case‑insensitive.

    is_palindrome(s):
        Return ``True`` if *s* reads the same forward and backward when
        case, spaces and punctuation are ignored.

    caesar_cipher(s, shift):
        Shift each alphabetical character in *s* by *shift* positions.
        Only letters ``a‑z`` and ``A‑Z`` are transformed; all other characters
        are left untouched.  ``shift`` may be negative.

    most_common_word(s):
        Return the most frequent word in *s* (case‑insensitive).  If several
        words share the highest frequency, the one that appears first in the
        original text is returned.  ``None`` is returned for an empty string.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in the given string.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            The words of *s* in reverse order, separated by a single space.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, ignoring case.

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
        Determine whether *s* is a palindrome, ignoring case, spaces and
        punctuation.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if *s* is a palindrome under the described rules.
        """
        # Keep only alphanumeric characters and lower‑case them
        cleaned = [ch.lower() for ch in s if ch.isalnum()]
        return cleaned == list(reversed(cleaned))

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* using the integer *shift*.

        Only the 26 English letters are shifted; their case is preserved.
        Non‑alphabetic characters are returned unchanged.  ``shift`` may be
        negative.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift each letter.

        Returns
        -------
        str
            The transformed string.
        """
        result = []

        # Normalise shift to the range [0, 25] for each alphabet cycle
        shift = shift % 26

        for ch in s:
            if 'a' <= ch <= 'z':
                # ord('a') = 97
                new_code = (ord(ch) - ord('a') + shift) % 26 + ord('a')
                result.append(chr(new_code))
            elif 'A' <= ch <= 'Z':
                # ord('A') = 65
                new_code = (ord(ch) - ord('A') + shift) % 26 + ord('A')
                result.append(chr(new_code))
            else:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in *s* (case‑insensitive).  If there is a
        tie, the word that appears first in the original string is chosen.
        Punctuation is ignored; words are delimited by whitespace.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word, or ``None`` if *s* is empty or contains no
            words.
        """
        if not s or not s.strip():
            return None

        # Extract words (alphanumeric + underscore) ignoring punctuation.
        # ``\\b\\w+\\b`` matches sequences of word characters bounded by word
        # boundaries.
        words = re.findall(r"\b\w+\b", s.lower())

        if not words:
            return None

        # Count frequencies
        counter = Counter(words)
        max_freq = max(counter.values())

        # Return the first word (in original lower‑cased order) with max_freq
        for w in words:
            if counter[w] == max_freq:
                return w  # lower‑cased version; callers can .capitalize() if needed
        # Unreachable – kept for type‑checkers
        return None

import pytest


def test_reverse_words():
    assert StringProcessor.reverse_words("the quick brown fox") == "fox brown quick the"
    assert StringProcessor.reverse_words("  hello   world  ") == "world hello"


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
    # Positive shift
    assert StringProcessor.caesar_cipher("abc XYZ", 3) == "def ABC"
    # Negative shift
    assert StringProcessor.caesar_cipher("def ABC", -3) == "abc XYZ"
    # Mixed characters unchanged
    assert StringProcessor.caesar_cipher("Hello, World! 123", 5) == "Mjqqt, Btwqi! 123"


def test_most_common_word():
    txt = "Apple banana apple orange banana apple"
    # "apple" appears 3 times, "banana" 2 times → "apple"
    assert StringProcessor.most_common_word(txt) == "apple"

    # Tie‑breaking: "cat" and "dog" both appear twice; "cat" comes first.
    txt2 = "Cat dog cat dog"
    assert StringProcessor.most_common_word(txt2) == "cat"

    # Empty string → None
    assert StringProcessor.most_common_word("") is None