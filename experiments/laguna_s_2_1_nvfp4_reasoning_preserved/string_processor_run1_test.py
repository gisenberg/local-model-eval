import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """
    A class to perform various string processing operations.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a given string.

        Args:
            s (str): The input string.

        Returns:
            str: The string with the order of words reversed.
        """
        return ' '.join(reversed(s.split()))

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a given string (case-insensitive).

        Args:
            s (str): The input string.

        Returns:
            int: The number of vowels in the string.
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a given string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s (str): The input string.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a given string.

        Only letters (a-z, A-Z) are shifted; other characters remain unchanged.
        Supports negative shifts.

        Args:
            s (str): The input string.
            shift (int): The number of positions to shift each letter.

        Returns:
            str: The encrypted/decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                base = ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif 'A' <= char <= 'Z':
                base = ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in a given string.

        The search is case-insensitive. If there is a tie, the word that appears
        first in the string is returned.

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

        # Find the first word with the maximum count
        for word in words:
            if word_counts[word] == max_count:
                return word
        return None

import pytest


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"


def test_count_vowels(processor):
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("AEIOUaeiou") == 10
    assert processor.count_vowels("The quick brown fox") == 5


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("No lemon, no melon") is True
    assert processor.is_palindrome("") is True
    assert processor.is_palindrome("a") is True


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 1) == "YZA"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert processor.caesar_cipher("Mjqqt, Btwqi!", -5) == "Hello, World!"


def test_most_common_word(processor):
    assert processor.most_common_word("Hello hello world") == "hello"
    assert processor.most_common_word("The the quick brown fox jumps over the lazy dog") == "the"
    assert processor.most_common_word("apple orange banana apple orange") == "apple"
    assert processor.most_common_word("One") == "one"
    assert processor.most_common_word("") is None