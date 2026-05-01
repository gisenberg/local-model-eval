"""Utility class for common string manipulations."""

from __future__ import annotations

from typing import Optional


class StringProcessor:
    """Collection of small string‑processing helpers."""

    def reverse_words(self, s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are delimited by any whitespace; the result is joined with a
        single space.

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

    def count_vowels(self, s: str) -> int:
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
        vowels = set("aeiou")
        return sum(1 for ch in s.lower() if ch in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Determine whether ``s`` is a palindrome, ignoring case,
        whitespace and punctuation.

        Only alphanumeric characters are considered.

        Parameters
        ----------
        s: str
            Input string.

        Returns
        -------
        bool
            ``True`` if the cleaned string reads the same forwards and backwards.
        """
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to alphabetic characters.

        The shift is applied to the alphabet (a‑z, A‑Z) and wraps around.
        Non‑alphabetic characters are left untouched.  Negative shifts are
        supported.

        Parameters
        ----------
        s: str
            Text to encode.
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

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most common word in ``s`` (case‑insensitive).

        Words are split on whitespace.  If several words share the highest
        frequency, the one that appears first in the original text is
        returned.  An empty or whitespace‑only string yields ``None``.

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

        words = s.lower().split()
        counts: dict[str, int] = {}
        order: list[str] = []  # preserves first occurrence order

        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if w not in counts:  # first time we see this word
                order.append(w)

        max_count = max(counts.values())
        for w in order:
            if counts[w] == max_count:
                return w
        return None  # never reached, kept for type‑checkers

import pytest



def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    # multiple spaces are collapsed by split()
    assert sp.reverse_words("  leading  spaces  ") == "spaces leading"


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3          # e, o, o
    assert sp.count_vowels("AEIOUaeiou") == 10
    assert sp.count_vowels("") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama")
    assert sp.is_palindrome("No lemon, no melon")
    assert not sp.is_palindrome("Hello, World!")


def test_caesar_cipher():
    sp = StringProcessor()
    # positive shift
    assert sp.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
    # negative shift (should decode back)
    assert sp.caesar_cipher("Ifmmp, Xpsme!", -1) == "Hello, World!"
    # preserve case and non‑letters
    assert sp.caesar_cipher("Abc-XYZ", 2) == "Cde-ZA"


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("The the cat and the dog") == "the"
    # all words appear once → first word wins
    assert sp.most_common_word("cat dog mouse") == "cat"
    # empty string → None
    assert sp.most_common_word("") is None
    # tie – first occurrence should be returned
    assert sp.most_common_word("apple banana apple banana") == "apple"