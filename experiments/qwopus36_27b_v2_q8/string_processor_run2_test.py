import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in the given string.
        Handles multiple spaces by collapsing them into a single space.
        """
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels (a, e, i, o, u) in the given string.
        The check is case-insensitive.
        """
        return sum(1 for char in s.lower() if char in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if the given string is a palindrome.
        Ignores case, spaces, and punctuation.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to the string, shifting only a-z and A-Z 
        characters by the specified shift amount. 
        Supports negative shifts and wraps around the alphabet.
        Non-alphabetic characters remain unchanged.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift with wrap-around; Python's modulo handles negatives correctly
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Returns the most common word in the string.
        The check is case-insensitive. If there is a tie, 
        it returns the first word encountered in the string.
        Returns None if the string is empty or contains no words.
        """
        # Extract words, removing punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
            
        counter = Counter(words)
        # Counter.most_common(1) returns the top item.
        # Since Python's sort is stable, tied counts preserve insertion order,
        # which corresponds to the first encountered word.
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None

import pytest

@pytest.fixture
def processor():
    """Creates a StringProcessor instance for tests."""
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("Hello World") == "World Hello"
    assert processor.reverse_words("  Python   is   great  ") == "great is Python"
    assert processor.reverse_words("Single") == "Single"

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU aeio") == 10
    assert processor.count_vowels("bcdfg") == 0

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("12321") is True

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 3) == "abc"  # wrap-around
    assert processor.caesar_cipher("abc", -1) == "zab" # negative shift
    assert processor.caesar_cipher("Hello!", 13) == "Uryyb!" # rot13

def test_most_common_word(processor):
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("a b a b") == "a"  # Tie -> first encountered
    assert processor.most_common_word("Hello, hello!") == "hello" # Punctuation ignored
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None