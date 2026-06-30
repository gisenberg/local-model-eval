import re
from typing import Optional

class StringProcessor:
    """A utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        clean = ''.join(c.lower() for c in s if c.isalnum())
        return clean == clean[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to the string, shifting only a-z/A-Z characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        Returns the first word encountered in case of a tie. Returns None for empty input."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        max_count = 0
        most_common = None
        
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if counts[w] > max_count:
                max_count = counts[w]
                most_common = w
                
        return most_common


# ==================== PYTEST TESTS ====================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  python is great  ") == "great is python"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("Test 123", 0) == "Test 123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat") == "cat"
    assert sp.most_common_word("tie tie") == "tie"
    assert sp.most_common_word("") is None
    # Test tie-breaking: first encountered wins
    assert sp.most_common_word("b a b a") == "b"