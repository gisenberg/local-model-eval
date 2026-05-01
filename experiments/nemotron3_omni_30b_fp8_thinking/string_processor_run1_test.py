from __future__ import annotations

import string
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    Utility class that provides a handful of string‑processing helpers.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are delimited by any whitespace (space, tab, newline, …).
        Consecutive whitespace characters are treated as a single separator
        and the result is joined with a single space.

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
            ``True`` if the cleaned string reads the same forward and backward.
        """
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* shifting alphabetic characters by *shift*.
        The shift works for both lower‑ and upper‑case letters and wraps around
        the alphabet. Non‑alphabetic characters are left unchanged.

        Parameters
        ----------
        s: str
            Text to be encoded.
        shift: int
            Number of positions to shift; can be negative.

        Returns
        -------
        str
            The encoded string.
        """
        result = []
        for ch in s:
            if "a" <= ch <= "z":
                base = ord("a")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            elif "A" <= ch <= "Z":
                base = ord("A")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most common word in *s* (case‑insensitive).  If several
        words share the highest frequency, the one that appears first in the
        original string is returned.  Returns ``None`` when *s* contains no
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
        if not s.strip():
            return None

        words = s.split()
        # Count frequencies (case‑insensitive)
        lowered = [w.lower() for w in words]
        freq = Counter(lowered)

        # Keep the first original appearance for each lower‑cased word
        first_occurrence: dict[str, tuple[int, str]] = {}
        for idx, word in enumerate(words):
            key = word.lower()
            if key not in first_occurrence:
                first_occurrence[key] = (idx, word)   # (position, original word)

        max_count = max(freq.values())
        # All words that share the max frequency
        candidates = [w for w, c in freq.items() if c == max_count]

        # Choose the one with the smallest original index
        chosen_key = min(candidates, key=lambda w: first_occurrence[w][0])
        return first_occurrence[chosen_key][1]


# -------------------------------------------------------------------------
# pytest test suite (5 tests)
# -------------------------------------------------------------------------

def test_reverse_words():
    assert StringProcessor().reverse_words("hello world") == "world hello"
    assert StringProcessor().reverse_words("  many   spaces  here  ") == "here spaces many"


def test_count_vowels():
    assert StringProcessor().count_vowels("Hello World") == 3   # e, o, o
    assert StringProcessor().count_vowels("") == 0
    assert StringProcessor().count_vowels("AEIOU") == 5


def test_is_palindrome():
    assert StringProcessor().is_palindrome("A man, a plan, a canal: Panama")
    assert StringProcessor().is_palindrome("RaceCar")
    assert not StringProcessor().is_palindrome("Hello, World!")


def test_caesar_cipher():
    assert StringProcessor().caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor().caesar_cipher("abc XYZ", -1) == "zab yza"
    # non‑letters stay the same
    assert StringProcessor().caesar_cipher("Hello, World! 123", 2) == "Jgnnq, Yqtng! 123"


def test_most_common_word():
    txt = "Apple apple banana banana apple"
    # "apple" appears 3 times, case‑insensitively, and is the first with max count
    assert StringProcessor().most_common_word(txt) == "Apple"

    # tie – first occurrence wins
    txt2 = "dog cat mouse cat dog"
    # both "dog" and "cat" appear twice; "dog" appears first
    assert StringProcessor().most_common_word(txt2) == "dog"

    assert StringProcessor().most_common_word("") is None