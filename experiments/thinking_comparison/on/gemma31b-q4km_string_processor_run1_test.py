import re
from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces and removing leading/trailing whitespace."""
        # .split() without arguments splits by any whitespace and discards empty strings
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels (a, e, i, o, u) in a string, case-insensitive."""
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = "".join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to a-z and A-Z characters, leaving others unchanged."""
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Calculate shift for lowercase
                start = ord('a')
                result.append(chr((ord(char) - start + shift) % 26 + start))
            elif 'A' <= char <= 'Z':
                # Calculate shift for uppercase
                start = ord('A')
                result.append(chr((ord(char) - start + shift) % 26 + start))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Returns the first occurrence in case of a tie."""
        if not s.strip():
            return None
        
        # Normalize to lowercase and split into words
        words = s.lower().split()
        if not words:
            return None
            
        counts = Counter(words)
        # most_common(1) returns a list containing a tuple: [(word, count)]
        # In Python 3.7+, Counter remembers insertion order, 
        # so most_common() handles ties by returning the first encountered element.
        return counts.most_common(1)[0][0]

# --- Pytest Tests ---
# To run these, save this file as test_processor.py and run `pytest test_processor.py`

import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is awesome") == "awesome is Python"
    assert sp.reverse_words("  single  ") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    # Test positive shift
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    # Test negative shift
    assert sp.caesar_cipher("Khoor, Zruog!", -3) == "Hello, World!"
    # Test wrap around (z -> a)
    assert sp.caesar_cipher("xyz", 1) == "yza"

def test_most_common_word():
    sp = StringProcessor()
    # Standard case
    assert sp.most_common_word("Apple banana apple orange banana apple") == "apple"
    # Tie case (should return first appearing: 'apple')
    assert sp.most_common_word("apple banana apple banana") == "apple"
    # Empty string case
    assert sp.most_common_word("   ") is None
    # Case insensitivity
    assert sp.most_common_word("Apple apple APPLE") == "apple"