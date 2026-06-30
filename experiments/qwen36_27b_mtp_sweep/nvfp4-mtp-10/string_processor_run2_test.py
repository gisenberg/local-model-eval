import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher shift to alphabetic characters.
        Non-alphabetic characters remain unchanged. Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Returns the most frequent word in the string (case-insensitive).
        Returns the first occurring word in case of a tie. Returns None if no words are found.
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
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
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd", -1) == "abc"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("123 !@#", 5) == "123 !@#"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange") == "apple"
    assert StringProcessor.most_common_word("cat dog cat") == "cat"
    assert StringProcessor.most_common_word("123 !@#") is None
    # Tie-breaking test: 'b' appears first and reaches max count first
    assert StringProcessor.most_common_word("b a b a") == "b"