import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are defined as substrings separated by whitespace.
        The original spacing between words is not preserved – a single
        space is used to join the reversed words.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        str
            String with word order reversed.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in the string,
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
        Determine whether *s* is a palindrome when case, spaces and
        punctuation are ignored.

        Only alphanumeric characters are considered.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            True if the cleaned string reads the same forwards and
            backwards, False otherwise.
        """
        cleaned = re.sub(r"[^A-Za-z0-9]", "", s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s*, shifting only alphabetic characters.
        Upper‑case and lower‑case letters are handled separately and wrap
        around the alphabet. Non‑letters are left unchanged.

        Parameters
        ----------
        s: str
            Input string.
        shift: int
            Number of positions to shift; can be negative.

        Returns
        -------
        str
            Encrypted/decrypted string.
        """
        def shift_char(ch: str) -> str:
            if 'a' <= ch <= 'z':
                base = ord('a')
                return chr((ord(ch) - base + shift) % 26 + base)
            if 'A' <= ch <= 'Z':
                base = ord('A')
                return chr((ord(ch) - base + shift) % 26 + base)
            return ch

        return "".join(shift_char(ch) for ch in s)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most frequent word in *s* (case‑insensitive).
        If several words share the highest frequency, the word that
        appears first in the original string is returned.
        If the string contains no words, ``None`` is returned.

        Words are sequences of alphanumeric characters and underscores
        (``\\w+``).

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        Optional[str]
            The most common word preserving its original casing from the
            first occurrence, or ``None`` if no words are present.
        """
        # Find words preserving original order
        words = re.findall(r"\w+", s)
        if not words:
            return None

        # Lower‑cased versions for counting
        lowered = [w.lower() for w in words]
        freq = Counter(lowered)

        # Determine highest frequency
        max_count = max(freq.values())

        # Return the first word (original case) whose lower‑cased form
        # reaches max_count
        for original, low in zip(words, lowered):
            if freq[low] == max_count:
                return original
        return None   # unreachable

import pytest


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  leading and trailing  ") == "trailing and leading"
    assert sp.reverse_words("") == ""
    assert sp.reverse_words("single") == "single"


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOUaeiou") == 10
    assert sp.count_vowels("Python") == 1
    assert sp.count_vowels("") == 0
    assert sp.count_vowels("123!@#") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("No lemon, no melon") is True
    assert sp.is_palindrome("Hello") is False
    assert sp.is_palindrome("") is True   # empty string is trivially palindrome
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("XYZ", 2) == "ZAB"
    assert sp.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert sp.caesar_cipher("Mjqqt, Btwqi!", -5) == "Hello, World!"
    assert sp.caesar_cipher("Python3", 0) == "Python3"


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("The quick brown fox jumps over the lazy dog") == "the"
    assert sp.most_common_word("apple banana apple orange banana") == "apple"
    assert sp.most_common_word("one two three") == "one"   # tie -> first
    assert sp.most_common_word("!!! ???") is None
    assert sp.most_common_word("") is None