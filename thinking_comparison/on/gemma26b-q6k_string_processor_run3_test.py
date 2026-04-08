import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for various string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string, collapsing multiple spaces."""
        # split() without arguments handles any whitespace and removes leading/trailing
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in a case-insensitive manner."""
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies Caesar cipher shift to letters, leaving other characters unchanged."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine if uppercase or lowercase base
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position using modulo 26 to handle wraps and negative shifts
                new_char = chr((ord(char) - start + shift) % 26 + start)
                result.append(new_char)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word (case-insensitive). Returns None for empty strings."""
        # Use regex to find all words (sequences of alphanumeric characters)
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
        
        # Counter in Python 3.7+ maintains insertion order. 
        # most_common(1) will return the first encountered word in case of a tie.
        counts = Counter(words)
        return counts.most_common(1)[0][0]


# --- Pytest Tests ---
# To run these, save this file as string_processor.py and run `pytest string_processor.py`

import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is fun") == "fun is Python"
    assert sp.reverse_words("Single") == "Single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOUaeiou") == 10
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race car") is True
    assert sp.is_palindrome("hello") is False

def test_caesar_cipher():
    sp = StringProcessor()
    # Test positive shift
    assert sp.caesar_cipher("Abc!", 1) == "Bcd!"
    # Test wrap around
    assert sp.caesar_cipher("xyz", 3) == "abc"
    # Test negative shift
    assert sp.caesar_cipher("Bcd!", -1) == "Abc!"
    # Test non-alpha characters
    assert sp.caesar_cipher("123! @#", 5) == "123! @#"

def test_most_common_word():
    sp = StringProcessor()
    # Standard case
    assert sp.most_common_word("Apple banana apple Banana cherry") == "apple"
    # Tie-breaker (should return 'a' because it appears first)
    assert sp.most_common_word("a b b a") == "a"
    # Empty/No words
    assert sp.most_common_word("") is None
    assert sp.most_common_word("!!!") is None