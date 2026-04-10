import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for various string manipulation tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words, collapsing multiple spaces and stripping edges."""
        # split() without arguments splits by any whitespace and discards empty strings
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
        """Applies Caesar cipher shift to alphabetic characters only."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine if uppercase or lowercase base
                start = ord('a') if char.islower() else ord('A')
                # Calculate new character position using modulo 26
                new_char = chr((ord(char) - start + shift) % 26 + start)
                result.append(new_char)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word (case-insensitive). Returns first if tied."""
        # Use regex to find words (alphanumeric sequences)
        words = re.findall(r'\w+', s.lower())
        
        if not words:
            return None
        
        counts = Counter(words)
        max_freq = max(counts.values())
        
        # To handle the tie-breaker (return the one that appears first),
        # we iterate through the original word list.
        for word in words:
            if counts[word] == max_freq:
                return word
        return None

# --- Pytest Tests ---

import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is awesome") == "awesome is Python"
    assert sp.reverse_words("   ") == ""

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
    assert sp.caesar_cipher("Abc!", 1) == "Bcd!"
    assert sp.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert sp.caesar_cipher("Z", 1) == "A"
    assert sp.caesar_cipher("A", -1) == "Z"

def test_most_common_word():
    sp = StringProcessor()
    # Standard case
    assert sp.most_common_word("Apple banana apple cherry") == "apple"
    # Tie-breaker (apple and banana both appear twice, apple is first)
    assert sp.most_common_word("apple banana apple banana") == "apple"
    # Empty/Whitespace case
    assert sp.most_common_word("   ") is None
    # Case insensitivity
    assert sp.most_common_word("Test test TEST") == "test"