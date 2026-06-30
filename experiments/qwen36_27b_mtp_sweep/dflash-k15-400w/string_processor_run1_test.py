import re
from typing import Optional

class StringProcessor:
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
        """Apply a Caesar cipher to the string, shifting only alphabetic characters.
        Supports negative shifts and preserves original case."""
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive).
        Returns the first word encountered if there is a tie.
        Returns None if the string contains no words."""
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
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def", -3) == "abc"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana") == "apple"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("cat dog cat dog") == "cat"