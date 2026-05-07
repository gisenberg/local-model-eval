import re
from typing import Optional
from collections import Counter

class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order. Multiple spaces are normalized.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Integer count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to a string, shifting only a-z/A-Z characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. If tied, returns the first one encountered.
            Returns None if the string contains no words.
        """
        if not s or not s.strip():
            return None
            
        # Extract words, ignoring punctuation and normalizing case
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Return first word in original order that matches the max frequency
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("") == ""

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True
    assert StringProcessor.is_palindrome("") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("xyz", -1) == "wxy"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("123 !@#", 5) == "123 !@#"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie: first encountered wins
    assert StringProcessor.most_common_word("  punctuation, everywhere! ") == "everywhere"
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None