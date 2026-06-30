import re
from typing import Optional
from collections import Counter

class StringProcessor:
    """A utility class for common string manipulation and analysis tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the given string.
        Consecutive whitespace is treated as a single separator.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        clean = ''.join(c.lower() for c in s if c.isalnum())
        return clean == clean[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters only.
        Preserves case and leaves non-alphabetic characters unchanged.
        Supports positive and negative shifts.
        """
        result = []
        for c in s:
            if c.isalpha():
                base = ord('A') if c.isupper() else ord('a')
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        Returns the first occurrence if there's a tie. Returns None if no words exist.
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
        
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Iterate in original order to respect "first if tied" rule
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ==================== PYTEST TESTS ====================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("bcdfg") == 0
    assert StringProcessor.count_vowels("") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("XYZ", -1) == "WXY"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("Test 123", 0) == "Test 123"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # tie -> first wins
    assert StringProcessor.most_common_word("123 !@#") is None
    assert StringProcessor.most_common_word("The the THE") == "the"