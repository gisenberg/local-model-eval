import re
import string
from typing import Optional
from collections import Counter

class StringProcessor:
    """Utility class for common string processing operations."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case.
        Non-alphabetic characters remain unchanged. Supports negative shifts."""
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
        """Return the most common word in the string (case-insensitive).
        If multiple words are tied for the highest frequency, returns the first one encountered.
        Returns None if the string contains no words."""
        # Extract words, normalize to lowercase, and strip punctuation
        words = [w.strip(string.punctuation) for w in s.lower().split()]
        words = [w for w in words if w]
        
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Return the first word that matches the maximum count
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
    assert StringProcessor.reverse_words("  python   is great  ") == "great is python"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU") == 5
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("xyz", -1) == "wxy"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("abc", 26) == "abc"  # Full rotation

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple cherry banana") == "apple"
    assert StringProcessor.most_common_word("one two three") == "one"
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("!!@@##") is None