import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order. Extra whitespace is normalized.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Number of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if palindrome, False otherwise.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (supports negative values).
            
        Returns:
            Encrypted/decrypted string. Non-alphabetic characters remain unchanged.
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
        """Find the most frequent word in the string.
        
        Args:
            s: Input string.
            
        Returns:
            The most common word in lowercase. Returns the first occurring word
            in case of a tie. Returns None if no words are found.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        counts = Counter(words)
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

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

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tied, first wins
    assert sp.most_common_word("   !@#   ") is None