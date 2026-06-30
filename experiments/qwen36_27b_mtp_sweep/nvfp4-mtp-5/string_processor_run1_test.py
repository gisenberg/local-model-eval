import re
from typing import Optional

class StringProcessor:
    """Utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        Multiple consecutive spaces are normalized to a single space.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to the string, shifting only a-z/A-Z characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive).
        Returns the first word encountered if there's a tie.
        Returns None if the string contains no words.
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        max_count = 0
        most_common: Optional[str] = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
        return most_common


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  python   is   great  ") == "great is python"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU") == 5

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 2) == "zab"
    assert sp.caesar_cipher("ABC", -1) == "ZAB"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tie, returns first encountered
    assert sp.most_common_word("") is None
    assert sp.most_common_word("Hello HELLO hello") == "hello"