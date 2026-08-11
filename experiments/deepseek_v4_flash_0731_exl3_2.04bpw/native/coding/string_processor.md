```python
from typing import Optional

class StringProcessor:
    """A utility class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Multiple spaces between words are collapsed to a single space,
        and leading/trailing whitespace is removed.

        Args:
            s: Input string.

        Returns:
            String with words in reversed order.
        """
        words = s.split()  # splits on any whitespace, collapses multiple spaces
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.

        Args:
            s: Input string.

        Returns:
            Number of vowels.
        """
        vowels = set('aeiou')
        return sum(1 for ch in s.lower() if ch in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.

        Only alphanumeric characters are considered.

        Args:
            s: Input string.

        Returns:
            True if the cleaned string reads the same forwards and backwards.
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher with the given shift to letters (a-z, A-Z).

        Non-letter characters are left unchanged. Negative shifts are supported.

        Args:
            s: Input string.
            shift: Number of positions to shift (can be negative).

        Returns:
            Encoded string.
        """
        result = []
        for ch in s:
            if ch.isalpha():
                base = ord('a') if ch.islower() else ord('A')
                # Compute new character with modulo 26
                new_char = chr((ord(ch) - base + shift) % 26 + base)
                result.append(new_char)
            else:
                result.append(ch)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).

        In case of a tie, the word that appears first in the string is returned.
        Returns None for empty strings.

        Args:
            s: Input string.

        Returns:
            Most common word or None if no words.
        """
        words = s.split()
        if not words:
            return None

        counts = {}
        max_count = 0
        max_word = None

        for word in words:
            lower_word = word.lower()
            counts[lower_word] = counts.get(lower_word, 0) + 1
            count = counts[lower_word]
            # Only update if strictly greater, preserving first occurrence on ties
            if count > max_count:
                max_count = count
                max_word = lower_word

        return max_word


# ---------- pytest tests ----------

import pytest

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("Hello world") == "world Hello"
    assert processor.reverse_words("  multiple   spaces   here ") == "here spaces multiple"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello World") == 3  # e, o, o
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("No 'x' in Nixon") is True
    assert processor.is_palindrome("hello") is False
    assert processor.is_palindrome("") is True  # empty string is palindrome

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("XYZ", 2) == "ZAB"
    assert processor.caesar_cipher("123", 5) == "123"  # non-letters unchanged
    assert processor.caesar_cipher("", 10) == ""

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("cat dog cat dog") == "cat"  # tie, first appears
    assert processor.most_common_word("The quick brown fox") == "the"  # case-insensitive
    assert processor.most_common_word("") is None
    assert processor.most_common_word("one two two three three three") == "three"
```