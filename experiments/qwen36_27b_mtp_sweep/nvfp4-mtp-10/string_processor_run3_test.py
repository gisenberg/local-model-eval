import re
from typing import Optional
import pytest

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with the given shift.
        Only affects a-z/A-Z characters. Supports negative shifts.
        Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        If multiple words share the highest frequency, returns the first one encountered."""
        words = s.split()
        if not words:
            return None

        counts: dict[str, int] = {}
        order: list[str] = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in counts:
                order.append(w_lower)
            counts[w_lower] = counts.get(w_lower, 0) + 1

        max_count = max(counts.values())
        for w in order:
            if counts[w] == max_count:
                return w
        return None


# ========================
# Pytest Test Suite
# ========================

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("abc XYZ", -1) == "zab WXY"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("123 @#$", 5) == "123 @#$"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tie-breaker: first encountered
    assert sp.most_common_word("Hello hello HELLO") == "hello"
    assert sp.most_common_word("") is None