import re
from typing import Optional

class StringProcessor:
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
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher shift to alphabetic characters only.
        
        Non-alphabetic characters remain unchanged. Supports negative shifts.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (positive or negative).
            
        Returns:
            Ciphered string.
        """
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        
        If multiple words tie for the highest frequency, returns the one that 
        appears first in the original string. Returns None if no words are found.
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word, or None.
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
            
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            
        max_count = max(counts.values())
        return next(w for w in words if counts[w] == max_count)


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU") == 5

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tie, first appearance wins
    assert sp.most_common_word("123 !@#") is None