import re
from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverses the order of words, normalizing spaces."""
        # .split() without arguments splits by any whitespace and discards empty strings
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Counts vowels (a, e, i, o, u) case-insensitively."""
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters and lowercase them
        cleaned = "".join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies Caesar cipher to alphabetic characters, preserving case."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine if we start at 'A' or 'a'
                start = ord('A') if char.isupper() else ord('a')
                # Shift within the 26-letter alphabet range
                shifted = (ord(char) - start + shift) % 26
                result.append(chr(start + shifted))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word (case-insensitive). Ties go to the first occurrence."""
        if not s.strip():
            return None
        
        # Normalize to lowercase and split into words
        words = s.lower().split()
        counts = Counter(words)
        
        # max() in Python is stable; it returns the first item encountered in case of ties
        return max(words, key=lambda w: counts[w])

# --- Pytest Tests ---
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  Hello   World  ") == "World Hello"
    assert sp.reverse_words("Python is awesome") == "awesome is Python"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Apple") == 2
    assert sp.count_vowels("xyz") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Abc 123", 1) == "Bcd 123"
    assert sp.caesar_cipher("Hello Z!", 1) == "Ifmmp A!"
    assert sp.caesar_cipher("Ifmmp", -1) == "Hello"

def test_most_common_word():
    sp = StringProcessor()
    # Test basic functionality
    assert sp.most_common_word("Apple banana apple orange") == "apple"
    # Test tie-breaker (first occurrence)
    assert sp.most_common_word("dog cat dog cat") == "dog"
    # Test empty string
    assert sp.most_common_word("   ") is None