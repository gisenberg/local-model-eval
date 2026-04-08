import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for various string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words, normalizing whitespace."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in a string, case-insensitive."""
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Remove all non-alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies Caesar cipher to alphabetic characters, leaving others unchanged."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine if uppercase or lowercase to find the base ASCII value
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position using modulo 26 to handle large/negative shifts
                new_char = chr(start + (ord(char) - start + shift) % 26)
                result.append(new_char)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word (case-insensitive). Returns None if empty."""
        # Use regex to find words (ignoring punctuation)
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts = Counter(words)
        # Counter.most_common(1) returns [(word, count)]
        # Because Counter preserves insertion order for ties in Python 3.7+, 
        # it naturally handles the "first appearance" requirement.
        return counts.most_common(1)[0][0]


# --- Pytest Tests ---

import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is fun") == "fun is Python"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("xyz", 3) == "abc"  # Wrap around
    assert sp.caesar_cipher("bcd", -1) == "abc"  # Negative shift

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("Apple banana apple Orange banana apple") == "apple"
    assert sp.most_common_word("One Two Three") == "one" # Tie: returns first
    assert sp.most_common_word("") is None
    assert sp.most_common_word("!!!") is None