from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class for processing strings with various text manipulation methods."""
    
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        """
        words = s.split()
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string (case-insensitive)."""
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z. Other characters remain unchanged.
        Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                shifted = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                result.append(shifted)
            elif 'A' <= char <= 'Z':
                shifted = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
                result.append(shifted)
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first.
        Returns None for empty strings.
        """
        if not s or not s.strip():
            return None
        
        words = s.split()
        if not words:
            return None
        
        # Count frequencies while preserving order of first appearance
        word_counts = {}
        for word in words:
            lower_word = word.lower()
            word_counts[lower_word] = word_counts.get(lower_word, 0) + 1
        
        max_count = max(word_counts.values())
        
        # Find first word with max count (preserving first occurrence)
        for word in words:
            if word_counts[word.lower()] == max_count:
                return word.lower()
        
        return None


# Pytest tests
import pytest


class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        # Basic reversal
        assert processor.reverse_words("hello world") == "world hello"
        # Multiple spaces normalized
        assert processor.reverse_words("hello   world   test") == "test world hello"
        # Leading/trailing spaces removed
        assert processor.reverse_words("  hello world  ") == "world hello"
        # Single word
        assert processor.reverse_words("single") == "single"
        # Empty and whitespace only
        assert processor.reverse_words("") == ""
        assert processor.reverse_words("   ") == ""
    
    def test_count_vowels(self):
        processor = StringProcessor()
        # Basic count
        assert processor.count_vowels("hello") == 2  # e, o
        # Case insensitive
        assert processor.count_vowels("HELLO") == 2
        assert processor.count_vowels("HeLLo") == 2
        # No vowels
        assert processor.count_vowels("rhythm") == 0
        # Empty string
        assert processor.count_vowels("") == 0
        # All vowels
        assert processor.count_vowels("aeiouAEIOU") == 10
    
    def test_is_palindrome(self):
        processor = StringProcessor()
        # Simple palindrome
        assert processor.is_palindrome("racecar") == True
        # Case insensitive
        assert processor.is_palindrome("RaceCar") == True
        # With spaces and punctuation
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        # Not a palindrome
        assert processor.is_palindrome("hello") == False
        # Empty string (technically palindrome)
        assert processor.is_palindrome("") == True
        # Single character
        assert processor.is_palindrome("a") == True
    
    def test_caesar_cipher(self):
        processor = StringProcessor()
        # Basic shift
        assert processor.caesar_cipher("abc", 1) == "bcd"
        # Wrap around lowercase
        assert processor.caesar_cipher("xyz", 1) == "yza"
        # Negative shift
        assert processor.caesar_cipher("bcd", -1) == "abc"
        # Mixed case
        assert processor.caesar_cipher("Abc", 1) == "Bcd"
        # Non-alphabetic characters unchanged
        assert processor.caesar_cipher("hello, world!", 1) == "ifmmp, xpsme!"
        # Large shift (should wrap)
        assert processor.caesar_cipher("abc", 26) == "abc"
        # Negative wrap around
        assert processor.caesar_cipher("abc", -1) == "zab"
    
    def test_most_common_word(self):
        processor = StringProcessor()
        # Basic case
        assert processor.most_common_word("hello world hello") == "hello"
        # Case insensitive counting
        assert processor.most_common_word("Hello world HELLO") == "hello"
        # Tie - first occurrence wins
        assert processor.most_common_word("a b a b") == "a"
        # Single word
        assert processor.most_common_word("single") == "single"
        # Empty string
        assert processor.most_common_word("") is None
        # Only spaces
        assert processor.most_common_word("   ") is None
        # All same word
        assert processor.most_common_word("the the the") == "the"