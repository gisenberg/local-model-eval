import re
from typing import Optional
import pytest

class StringProcessor:
    """A utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in the given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            A new string with the words in reverse order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            The total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies a Caesar cipher to the string, shifting only a-z/A-Z characters.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            The transformed string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most common word in the string (case-insensitive).
        
        Args:
            s: Input string containing words.
            
        Returns:
            The most frequent word. Returns the first encountered word if tied.
            Returns None if the string is empty or contains no words.
        """
        words = s.lower().split()
        if not words:
            return None
            
        counts = {}
        max_count = 0
        most_common = None
        
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
                
        return most_common


# ========================
# Pytest Test Suite
# ========================

@pytest.fixture
def sp():
    return StringProcessor()

def test_reverse_words(sp):
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels(sp):
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome(sp):
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher(sp):
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word(sp):
    assert sp.most_common_word("apple banana apple orange banana") == "apple"
    assert sp.most_common_word("test") == "test"
    assert sp.most_common_word("") is None