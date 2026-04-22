import re
from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters, preserving case and non-alphabetic characters."""
        result = []
        for char in s:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        Returns the first encountered word if there's a tie. Returns None if no words."""
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None
        # Counter.most_common() preserves insertion order for ties in Python 3.7+
        return Counter(words).most_common(1)[0][0]


# ========================
# Pytest Tests
# ========================
import pytest

class TestStringProcessor:
    @pytest.fixture
    def sp(self):
        return StringProcessor()

    def test_reverse_words(self, sp):
        assert sp.reverse_words("hello world") == "world hello"
        assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"

    def test_count_vowels(self, sp):
        assert sp.count_vowels("Hello World") == 3
        assert sp.count_vowels("rhythm") == 0
        assert sp.count_vowels("AEIOU") == 5

    def test_is_palindrome(self, sp):
        assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
        assert sp.is_palindrome("racecar") is True
        assert sp.is_palindrome("hello") is False

    def test_caesar_cipher(self, sp):
        assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
        assert sp.caesar_cipher("xyz", -1) == "wxy"
        assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

    def test_most_common_word(self, sp):
        assert sp.most_common_word("apple banana apple orange banana") == "apple"
        assert sp.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
        assert sp.most_common_word("") is None