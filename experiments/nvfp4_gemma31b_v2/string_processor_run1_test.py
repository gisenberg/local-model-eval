import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for processing and manipulating strings."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) case-insensitively."""
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies Caesar cipher to letters. 
        Preserves case and non-alphabetic characters. Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position within 0-25 range
                shifted = (ord(char) - start + shift) % 26
                result.append(chr(start + shifted))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Returns the most frequent word (case-insensitive). 
        Returns the first occurring word in case of a tie.
        """
        if not s.strip():
            return None
        
        # Remove punctuation and split by whitespace
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_freq = max(counts.values())
        
        # Find the first word in the original sequence that has the max frequency
        for word in words:
            if counts[word] == max_freq:
                return word
        return None

# --- Pytest Tests ---
# To run these, save as test_string_processor.py and run `pytest`
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  Python is fun  ") == "fun is Python"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("xyz") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Abc", 1) == "Bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("Hello!", -1) == "Gdkkn!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("The cat sat on the mat") == "the"
    assert sp.most_common_word("Apple banana apple banana") == "apple" # Tie: first one wins
    assert sp.most_common_word("") is None