import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in a string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case and non-alphabetic characters."""
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted)
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in a string (case-insensitive). 
        Returns the first encountered word if there's a tie."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        best_word = None
        max_count = 0
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                best_word = word
        return best_word


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("Rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("XYZ", -1) == "WXY"
    assert sp.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    # Tie-breaking: 'cat' appears first, so it should be returned
    assert sp.most_common_word("cat dog cat bird dog") == "cat"
    assert sp.most_common_word("  ... ,,, ") is None