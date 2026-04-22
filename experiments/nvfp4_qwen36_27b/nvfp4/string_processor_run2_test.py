import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in the string.
        Multiple consecutive whitespace characters are collapsed to a single space.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher shift to alphabetic characters only.
        Supports positive and negative shifts. Non-alphabetic characters remain unchanged.
        """
        result = []
        for c in s:
            if 'a' <= c <= 'z':
                result.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= c <= 'Z':
                result.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(c)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Finds the most common word in the string (case-insensitive).
        Returns the first occurrence if multiple words are tied for the highest count.
        Returns None if the string contains no words.
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
        counts = Counter(words)
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Test Suite
# ========================

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  python   is  great  ") == "great is python"
    assert StringProcessor.reverse_words("") == ""

def test_count_vowels():
    assert StringProcessor.count_vowels("Programming") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("rhythm") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("Race car!") is True
    assert StringProcessor.is_palindrome("Hello, World!") is False

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # first if tied
    assert StringProcessor.most_common_word("123 !@#") is None