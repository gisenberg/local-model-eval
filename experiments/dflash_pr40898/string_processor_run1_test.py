import re
from typing import Optional

class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order. Leading/trailing whitespace 
            and multiple internal spaces are normalized.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Integer count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, 
            False otherwise.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters, preserving case.
        
        Non-alphabetic characters remain unchanged. Supports positive and 
        negative shifts with proper alphabet wrapping.
        
        Args:
            s: Input string.
            shift: Number of positions to shift letters.
            
        Returns:
            Ciphered string.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        
        If multiple words are tied for the highest frequency, returns the 
        first one encountered. Punctuation is ignored when extracting words.
        
        Args:
            s: Input string.
            
        Returns:
            The most common word, or None if the string contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
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

import pytest

@pytest.fixture
def sp():
    return StringProcessor()

def test_reverse_words(sp):
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels(sp):
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("Rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome(sp):
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("Hello") is False

def test_caesar_cipher(sp):
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert sp.caesar_cipher("a-z 123!", 1) == "b-a 123!"

def test_most_common_word(sp):
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tied, returns first
    assert sp.most_common_word("... !!!") is None