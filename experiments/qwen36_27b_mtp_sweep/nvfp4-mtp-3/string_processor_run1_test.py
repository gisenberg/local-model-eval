import re
from typing import Optional

class StringProcessor:
    """Utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string containing words.
            
        Returns:
            String with words in reversed order. Multiple spaces are collapsed.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case. Supports negative shifts.
        
        Args:
            s: Input string.
            shift: Number of positions to shift. Can be negative.
            
        Returns:
            Ciphered string. Non-alphabetic characters remain unchanged.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                shifted = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted)
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. Returns the first encountered word if tied.
            Returns None if no words are found.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            
        max_count = max(counts.values())
        # Python 3.7+ dicts preserve insertion order, ensuring "first if tied" behavior
        for w, c in counts.items():
            if c == max_count:
                return w


# ========================
# Pytest Test Suite
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
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("a", -27) == "a"  # -27 ≡ -1 mod 26

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tie: first encountered wins
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None