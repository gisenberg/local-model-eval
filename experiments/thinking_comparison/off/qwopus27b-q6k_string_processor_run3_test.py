from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A utility class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading and trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with word order reversed
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.
        
        Args:
            s: Input string
            
        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome.
        
        Ignores case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z. Other characters remain unchanged.
        Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)
            
        Returns:
            Encrypted string
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(ord('a') + shifted))
            elif 'A' <= char <= 'Z':
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(ord('A') + shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first in the string.
        Returns None for empty strings or strings with only whitespace.
        
        Args:
            s: Input string
            
        Returns:
            Most common word in lowercase, or None if empty
        """
        if not s or not s.strip():
            return None
        
        words = s.split()
        if not words:
            return None
        
        word_counts = {}
        
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        max_count = max(word_counts.values())
        
        # Return first word with max count (preserving original order)
        for word in words:
            if word_counts[word.lower()] == max_count:
                return word.lower()
        
        return None


# Pytest tests
import pytest


class TestStringProcessor:
    def test_reverse_words(self):
        """Test reverse_words method."""
        processor = StringProcessor()
        
        # Basic reversal
        assert processor.reverse_words("hello world") == "world hello"
        
        # Multiple spaces become single space
        assert processor.reverse_words("hello   world") == "world hello"
        
        # Leading/trailing spaces removed
        assert processor.reverse_words("  hello world  ") == "world hello"
        
        # Single word
        assert processor.reverse_words("single") == "single"
        
        # Empty string
        assert processor.reverse_words("") == ""
        
        # Only spaces
        assert processor.reverse_words("   ") == ""

    def test_count_vowels(self):
        """Test count_vowels method."""
        processor = StringProcessor()
        
        # Basic count
        assert processor.count_vowels("hello") == 2  # e, o
        
        # Case insensitive
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("aeiou") == 5
        
        # No vowels
        assert processor.count_vowels("bcdfg") == 0
        
        # Mixed case
        assert processor.count_vowels("Hello World") == 3  # e, o, o
        
        # Empty string
        assert processor.count_vowels("") == 0

    def test_is_palindrome(self):
        """Test is_palindrome method."""
        processor = StringProcessor()
        
        # Simple palindrome
        assert processor.is_palindrome("racecar") == True
        
        # With punctuation and spaces
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        
        # Not a palindrome
        assert processor.is_palindrome("hello") == False
        
        # Case insensitive
        assert processor.is_palindrome("Was it a car or a cat I saw?") == True
        
        # Empty string is palindrome
        assert processor.is_palindrome("") == True
        
        # Single character
        assert processor.is_palindrome("a") == True

    def test_caesar_cipher(self):
        """Test caesar_cipher method."""
        processor = StringProcessor()
        
        # Basic positive shift
        assert processor.caesar_cipher("abc", 1) == "bcd"
        
        # Negative shift
        assert processor.caesar_cipher("abc", -1) == "zab"
        
        # Mixed case
        assert processor.caesar_cipher("Hello", 1) == "Ifmmp"
        
        # Non-alphabetic characters unchanged
        assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
        
        # Wrapping (shift by 26 = no change)
        assert processor.caesar_cipher("abc", 26) == "abc"
        
        # Large negative shift
        assert processor.caesar_cipher("abc", -26) == "abc"
        
        # Shift by 27 (same as 1)
        assert processor.caesar_cipher("abc", 27) == "bcd"

    def test_most_common_word(self):
        """Test most_common_word method."""
        processor = StringProcessor()
        
        # Basic case
        assert processor.most_common_word("the cat and the dog") == "the"
        
        # Case insensitive
        assert processor.most_common_word("The CAT and the dog") == "the"
        
        # Tie - first one wins
        assert processor.most_common_word("one two one two") == "one"
        
        # Empty string
        assert processor.most_common_word("") is None
        
        # Only whitespace
        assert processor.most_common_word("   ") is None
        
        # Single word
        assert processor.most_common_word("single") == "single"
        
        # Multiple ties - first occurrence wins
        assert processor.most_common_word("a b c a b c") == "a"