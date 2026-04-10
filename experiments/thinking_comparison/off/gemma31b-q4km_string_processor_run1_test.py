import re
from collections import Counter
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome, ignoring case, spaces, and punctuation."""
        cleaned = "".join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, supporting negative shifts."""
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Use modulo 26 to handle shifts larger than 26 and negative shifts
                shifted = chr(start + (ord(char) - start + shift) % 26)
                result.append(shifted)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Returns None if empty."""
        words = s.lower().split()
        if not words:
            return None
        
        counts = Counter(words)
        # max() in Python is stable, so it returns the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])

import pytest


@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    # Tests multiple spaces, leading/trailing spaces, and word reversal
    assert processor.reverse_words("  hello world  ") == "world hello"
    assert processor.reverse_words("The quick brown fox") == "fox brown quick The"
    assert processor.reverse_words("   multiple   spaces   here  ") == "here spaces multiple"

def test_count_vowels(processor):
    # Tests case insensitivity and strings with no vowels
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU aeiou") == 10
    assert processor.count_vowels("bcdfgh") == 0

def test_is_palindrome(processor):
    # Tests punctuation, case, and non-palindromes
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    # Tests positive shift, negative shift, and non-alpha characters
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("xyz", 3) == "abc"

def test_most_common_word(processor):
    # Tests case insensitivity, ties (first occurrence), and empty strings
    assert processor.most_common_word("Apple apple Banana apple banana") == "apple"
    assert processor.most_common_word("Dog cat dog cat") == "dog"  # Tie: dog appears first
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None