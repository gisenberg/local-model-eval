import re
from typing import Optional
from collections import Counter

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order. Extra whitespace is normalized.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Integer count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters only. Supports negative shifts.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (positive or negative).
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. Returns the first encountered word if tied.
            Returns None if the string contains no words.
        """
        # Extract alphabetic words only
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None
        # Counter.most_common() preserves insertion order for ties in Python 3.7+
        return Counter(words).most_common(1)[0][0]


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
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana") == "apple"
    assert StringProcessor.most_common_word("Hello world hello") == "hello"
    assert StringProcessor.most_common_word("123 !@#") is None