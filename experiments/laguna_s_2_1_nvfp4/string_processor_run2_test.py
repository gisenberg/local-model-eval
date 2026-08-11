from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class to process strings with various utility methods."""

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string.

        Args:
            s (str): The input string.

        Returns:
            str: A new string with the words in reverse order.
        """
        return ' '.join(reversed(s.split()))

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        Args:
            s (str): The input string.

        Returns:
            int: The number of vowels in the string.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s (str): The input string.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string, shifting only letters by a given amount.

        Args:
            s (str): The input string.
            shift (int): The number of positions to shift each letter.

        Returns:
            str: The encrypted/decrypted string.
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
        Finds the most common word in a string (case-insensitive).

        If there's a tie, returns the first occurring word among those tied.

        Args:
            s (str): The input string.

        Returns:
            Optional[str]: The most common word, or None if the string is empty.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        for word in words:
            if word_counts[word] == max_count:
                return word
        return None

import pytest


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("hello world here") == "here world hello"
    assert processor.reverse_words("a good     space") == "space good a"


def test_count_vowels(processor):
    assert processor.count_vowels("hello world") == 3
    assert processor.count_vowels("AEIOUaeiou") == 10


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"
    assert processor.caesar_cipher("abc", -1) == "zab"


def test_most_common_word(processor):
    assert processor.most_common_word("Bob hit a ball, the hit BALL flew") == "ball"
    assert processor.most_common_word("a a b b") == "a"
    assert processor.most_common_word("") is None