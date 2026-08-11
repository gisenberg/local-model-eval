from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class to process strings with various utility functions."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s (str): Input string.

        Returns:
            str: String with reversed word order.
        """
        return ' '.join(reversed(s.split()))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (case-insensitive) in a string.

        Args:
            s (str): Input string.

        Returns:
            int: Number of vowels.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s (str): Input string.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to the string, shifting only letters by `shift`.

        Args:
            s (str): Input string.
            shift (int): Shift amount (can be negative).

        Returns:
            str: Encrypted/decrypted string.
        """
        result = []

        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)

        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).
        If there's a tie, returns the first occurring one.

        Args:
            s (str): Input string.

        Returns:
            Optional[str]: The most common word, or None if no words exist.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        counter = Counter(words)
        max_count = max(counter.values())
        for word in words:
            if counter[word] == max_count:
                return word

import pytest


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"


def test_count_vowels(processor):
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("xyz") == 0


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("") is True


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 1) == "YZA"
    assert processor.caesar_cipher("abc", -1) == "zab"


def test_most_common_word(processor):
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("Hello hello HELLO") == "hello"
    assert processor.most_common_word("a a b b") == "a"
    assert processor.most_common_word("") is None