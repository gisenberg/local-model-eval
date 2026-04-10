from typing import Optional
import re


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
        # Split by whitespace, filter empty strings, reverse, join
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
        # Keep only alphanumeric characters, convert to lowercase
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift
            
        Returns:
            Encrypted string
        """
        result = []
        
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                offset = ord('a')
                shifted = (ord(char) - offset + shift) % 26 + offset
                result.append(chr(shifted))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                offset = ord('A')
                shifted = (ord(char) - offset + shift) % 26 + offset
                result.append(chr(shifted))
            else:
                # Leave other characters unchanged
                result.append(char)
        
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None
        
        # Split into words
        words = s.split()
        
        if not words:
            return None
        
        # Count frequencies (case-insensitive)
        freq = {}
        first_occurrence = {}
        
        for idx, word in enumerate(words):
            word_lower = word.lower()
            freq[word_lower] = freq.get(word_lower, 0) + 1
            if word_lower not in first_occurrence:
                first_occurrence[word_lower] = idx
        
        # Find most common (first in case of tie)
        max_count = max(freq.values())
        candidates = [w for w, c in freq.items() if c == max_count]
        
        # Return the one with earliest first occurrence
        return min(candidates, key=lambda w: first_occurrence[w])


# ==================== TESTS ====================

import pytest


class TestStringProcessor:
    """Test suite for StringProcessor class."""

    @pytest.fixture
    def processor(self):
        return StringProcessor()

    def test_reverse_words(self, processor):
        """Test reverse_words method."""
        # Basic case
        assert processor.reverse_words("hello world") == "world hello"
        
        # Multiple spaces
        assert processor.reverse_words("hello   world") == "world hello"
        
        # Leading/trailing spaces
        assert processor.reverse_words("  hello world  ") == "world hello"
        
        # Single word
        assert processor.reverse_words("single") == "single"
        
        # Empty string
        assert processor.reverse_words("") == ""
        
        # Multiple spaces between words
        assert processor.reverse_words("a   b    c") == "c b a"

    def test_count_vowels(self, processor):
        """Test count_vowels method."""
        # Basic case
        assert processor.count_vowels("hello") == 2  # e, o
        
        # Case insensitive
        assert processor.count_vowels("HELLO") == 2
        
        # Mixed case
        assert processor.count_vowels("HeLLo WoRLd") == 5
        
        # No vowels
        assert processor.count_vowels("rhythm") == 0
        
        # Empty string
        assert processor.count_vowels("") == 0
        
        # All vowels
        assert processor.count_vowels("aeiouAEIOU") == 10

    def test_is_palindrome(self, processor):
        """Test is_palindrome method."""
        # Simple palindrome
        assert processor.is_palindrome("racecar") == True
        
        # Palindrome with spaces
        assert processor.is_palindrome("A man a plan a canal Panama") == True
        
        # Palindrome with punctuation
        assert processor.is_palindrome("Was it a car or a cat I saw?") == True
        
        # Not a palindrome
        assert processor.is_palindrome("hello") == False
        
        # Empty string is palindrome
        assert processor.is_palindrome("") == True
        
        # Single character
        assert processor.is_palindrome("a") == True

    def test_caesar_cipher(self, processor):
        """Test caesar_cipher method."""
        # Basic shift
        assert processor.caesar_cipher("abc", 1) == "bcd"
        
        # Wrap around
        assert processor.caesar_cipher("xyz", 1) == "yza"
        
        # Negative shift
        assert processor.caesar_cipher("bcd", -1) == "abc"
        
        # Preserve case
        assert processor.caesar_cipher("AbC", 1) == "BcD"
        
        # Non-alphabetic characters unchanged
        assert processor.caesar_cipher("Hello, World! 123", 3) == "Khoor, Zruog! 123"
        
        # Large shift (wraps multiple times)
        assert processor.caesar_cipher("a", 27) == "b"
        
        # Zero shift
        assert processor.caesar_cipher("test", 0) == "test"

    def test_most_common_word(self, processor):
        """Test most_common_word method."""
        # Basic case
        assert processor.most_common_word("the cat and the dog") == "the"
        
        # Case insensitive
        assert processor.most_common_word("The THE the") == "the"
        
        # Tie - returns first occurrence
        assert processor.most_common_word("a b a b") == "a"
        
        # Single word
        assert processor.most_common_word("single") == "single"
        
        # Empty string returns None
        assert processor.most_common_word("") is None
        
        # Only whitespace returns None
        assert processor.most_common_word("   ") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])