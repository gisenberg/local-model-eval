import re
from typing import Optional


class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in the given string.
        Multiple consecutive whitespace characters are treated as a single delimiter.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels (a, e, i, o, u) in the string.
        Case-insensitive.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Checks if the string is a palindrome.
        Ignores case, spaces, and punctuation.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to alphabetic characters in the string.
        Preserves case and leaves non-alphabetic characters unchanged.
        Supports positive and negative shifts.
        """
        shift %= 26
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Returns the most frequently occurring word in the string.
        Case-insensitive. In case of a tie, returns the word that appears first.
        Returns None if the string contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        max_count = 0
        most_common = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word

        return most_common


# ========================
# Pytest Test Suite
# ========================
import pytest


def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("") == ""


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU") == 5


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("Was it a car or a cat I saw?") is True


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("xyz", 3) == "abc"
    assert StringProcessor.caesar_cipher("Hello, World!", -1) == "Gdkkn, Vnqkc!"
    assert StringProcessor.caesar_cipher("Test 123", 26) == "Test 123"


def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat") == "cat"
    assert StringProcessor.most_common_word("a b c") == "a"  # Tie scenario: first wins
    assert StringProcessor.most_common_word("   ") is None