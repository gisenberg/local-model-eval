from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class for processing and analyzing strings."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string.

        Returns:
            String with words in reversed order.

        Example:
            >>> StringProcessor.reverse_words("hello world")
            'world hello'
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in a string, case-insensitive.

        Args:
            s: Input string.

        Returns:
            Number of vowels.

        Example:
            >>> StringProcessor.count_vowels("Hello World")
            3
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string.

        Returns:
            True if the cleaned string is a palindrome, False otherwise.

        Example:
            >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
            True
        """
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to a string, shifting only alphabetic characters.

        Supports negative shifts and wraps around the alphabet.

        Args:
            s: Input string.
            shift: Integer shift value (can be negative).

        Returns:
            Encrypted string with shifted letters.

        Example:
            >>> StringProcessor.caesar_cipher("Hello", 3)
            'Khoor'
            >>> StringProcessor.caesar_cipher("Khoor", -3)
            'Hello'
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Find the most common word in a string, case-insensitive.

        Words are split by whitespace and punctuation is ignored.
        In case of a tie, the first occurring word is returned.

        Args:
            s: Input string.

        Returns:
            The most common word, or None if the string is empty or has no words.

        Example:
            >>> StringProcessor.most_common_word("apple banana apple cherry banana apple")
            'apple'
        """
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
        counts = Counter(words)
        # Find the first word with the maximum count (preserves order of first occurrence)
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# Pytest tests
import pytest


def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("one") == "one"
    assert StringProcessor.reverse_words("") == ""
    assert StringProcessor.reverse_words("  a  b  c  ") == "c b a"
    assert StringProcessor.reverse_words("Python is fun") == "fun is Python"


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("xyz") == 0
    assert StringProcessor.count_vowels("AEIOU") == 5
    assert StringProcessor.count_vowels("") == 0
    assert StringProcessor.count_vowels("Python Programming") == 4


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("") is True
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Hello", 3) == "Khoor"
    assert StringProcessor.caesar_cipher("Khoor", -3) == "Hello"
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("XYZ", 3) == "ABC"
    assert StringProcessor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"


def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple cherry banana apple") == "apple"
    assert StringProcessor.most_common_word("dog cat dog cat") == "dog"
    assert StringProcessor.most_common_word("a") == "a"
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("Hello, hello, HELLO!") == "hello"