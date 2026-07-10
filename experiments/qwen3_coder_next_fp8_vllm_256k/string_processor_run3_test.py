import string
from typing import Optional
from collections import Counter
import re


class StringProcessor:
    """A utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Words are defined as sequences of non-whitespace characters,
        separated by any whitespace.

        Args:
            s: Input string.

        Returns:
            A string with the words in reverse order.
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in a string (case-insensitive).

        Args:
            s: Input string.

        Returns:
            The count of vowels in the string.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Determine if a string is a palindrome, ignoring case, spaces, and punctuation.

        A palindrome reads the same forwards and backwards.

        Args:
            s: Input string.

        Returns:
            True if the string is a palindrome (after cleaning), False otherwise.
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to a string, shifting letters by the given amount.

        Only shifts alphabetic characters (a-z, A-Z); preserves case and non-letters.
        Negative shifts supported (shift right = shift left by negative).

        Args:
            s: Input string.
            shift: Number of positions to shift letters (can be negative).

        Returns:
            The encrypted string with letters shifted.
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]

        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)

        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).

        Words are sequences of alphanumeric characters.
        Returns the first word in case of a tie.

        Args:
            s: Input string.

        Returns:
            The most common word, or None if no words are found.
        """
        # Extract words (alphanumeric sequences), convert to lowercase
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None

        counter = Counter(words)
        max_count = max(counter.values())
        
        # Find first word (by original order) with max count
        for word in words:
            if counter[word] == max_count:
                return word


# === Pytest Tests ===

import pytest


class TestStringProcessor:
    @pytest.fixture
    def processor(self):
        return StringProcessor()

    def test_reverse_words_basic(self, processor):
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  multiple   spaces  here  ") == "here spaces multiple"

    def test_count_vowels_case_insensitive(self, processor):
        assert processor.count_vowels("Hello World") == 3
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("bcdfg") == 0
        assert processor.count_vowels("") == 0

    def test_is_palindrome(self, processor):
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert processor.is_palindrome("No lemon, no melon") is True
        assert processor.is_palindrome("Hello, world!") is False
        assert processor.is_palindrome("") is True  # empty string is palindrome
        assert processor.is_palindrome("racecar") is True

    def test_caesar_cipher_shifts(self, processor):
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("xyz", 3) == "abc"
        assert processor.caesar_cipher("Abc", -1) == "Zab"
        assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
        assert processor.caesar_cipher("test", 26) == "test"  # full rotation

    def test_most_common_word_ties_and_edge_cases(self, processor):
        assert processor.most_common_word("apple banana apple") == "apple"
        assert processor.most_common_word("a b a b") == "a"  # tie, first wins
        assert processor.most_common_word("Hello hello HELLO") == "hello"
        assert processor.most_common_word("!!!") is None  # no words
        assert processor.most_common_word("") is None