import re
from typing import Optional
from collections import Counter

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the given string.
        Multiple consecutive spaces are collapsed into a single space."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s if char.lower() in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to the string. Only affects a-z and A-Z. Supports negative shifts."""
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
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
        """Find the most common word in the string (case-insensitive).
        Returns the first occurrence if multiple words are tied."""
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        counts = Counter(words)
        max_count = max(counts.values())

        # Return first word in original order that matches the max count
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("bcdfg") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("12321") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert StringProcessor.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("z", -1) == "y"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie-break: first encountered
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None