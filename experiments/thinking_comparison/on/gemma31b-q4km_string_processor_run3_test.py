import string
from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces."""
        # .split() without arguments splits by any whitespace and discards empty strings
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels (a, e, i, o, u) case-insensitively."""
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = "".join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case."""
        result = []
        for char in s:
            if char.isupper():
                # Shift within A-Z (ASCII 65-90)
                result.append(chr((ord(char) - 65 + shift) % 26 + 65))
            elif char.islower():
                # Shift within a-z (ASCII 97-122)
                result.append(chr((ord(char) - 97 + shift) % 26 + 97))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Ties return the first occurrence."""
        if not s.strip():
            return None
        
        # Normalize to lowercase and split into words
        words = s.lower().split()
        if not words:
            return None
            
        counts = Counter(words)
        # max() in Python returns the first occurrence in case of a tie
        return max(words, key=lambda w: counts[w])

# --- Pytest Tests ---

import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("  hello   world  ") == "world hello"
    assert processor.reverse_words("Python is awesome") == "awesome is Python"
    assert processor.reverse_words("   ") == ""

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("bcdfg") == 0
    assert processor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    # Test positive shift
    assert processor.caesar_cipher("Hello Z!", 1) == "Ifmmp A!"
    # Test negative shift
    assert processor.caesar_cipher("Ifmmp A!", -1) == "Hello Z!"
    # Test large shift (wrap around)
    assert processor.caesar_cipher("abc", 28) == "cde"

def test_most_common_word(processor):
    # Test basic frequency
    assert processor.most_common_word("Apple banana apple orange banana apple") == "apple"
    # Test tie-breaker (should return 'apple' as it appears first)
    assert processor.most_common_word("apple banana apple banana") == "apple"
    # Test empty string
    assert processor.most_common_word("") is None
    # Test case insensitivity
    assert processor.most_common_word("Apple apple APPLE") == "apple"