import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for advanced string manipulation and analysis."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in a string (case-insensitive)."""
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Remove non-alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies Caesar cipher to a-z/A-Z characters. Non-alpha characters remain unchanged."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine if base is 'A' or 'a'
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position with wrap-around logic
                new_char = chr(start + (ord(char) - start + shift) % 26)
                result.append(new_char)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word in a string (case-insensitive). 
        Returns None if string is empty or contains no words.
        """
        # Use regex to find words (alphanumeric sequences)
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
        
        counts = Counter(words)
        # most_common(1) returns [(word, count)]
        return counts.most_common(1)[0][0]

# --- Pytest Suite ---
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  Python is  fun  ") == "fun is Python"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOUaeiou") == 10
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    # Positive shift
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    # Negative shift
    assert sp.caesar_cipher("bcd", -1) == "abc"
    # Non-alpha preservation
    assert sp.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("The cat and the dog") == "the"
    assert sp.most_common_word("Apple banana apple orange") == "apple"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("!!!") is None