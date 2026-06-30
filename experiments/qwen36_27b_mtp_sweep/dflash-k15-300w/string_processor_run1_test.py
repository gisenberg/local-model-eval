import re
from typing import Optional

class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher shift to alphabetic characters.
        
        Preserves case and leaves non-alphabetic characters unchanged.
        Supports positive and negative shifts.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift (can be negative).
            
        Returns:
            Ciphered string.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        
        If multiple words share the highest frequency, returns the first one 
        encountered in the original string.
        
        Args:
            s: Input text to analyze.
            
        Returns:
            The most common word, or None if the string contains no words.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        # Explicitly iterate through original order to guarantee "first encountered" tie-breaking
        for word in words:
            if counts[word] == max_count:
                return word
        return None

import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tied, first encountered
    assert sp.most_common_word("  !@#  ") is None