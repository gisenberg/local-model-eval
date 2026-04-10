from typing import Optional
from collections import Counter


class StringProcessor:
    """A class for various string processing operations."""
    
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with words in reversed order
        """
        words = s.split()  # split() handles multiple spaces and strips
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive.
        
        Args:
            s: Input string
            
        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome.
        
        Ignores case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        # Filter to alphanumeric only and convert to lowercase
        filtered = ''.join(c.lower() for c in s if c.isalnum())
        return filtered == filtered[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift.
        
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
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(ord('a') + shifted))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(ord('A') + shifted))
            else:
                # Leave other characters unchanged
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None
            
        words = s.split()
        if not words:
            return None
            
        # Count words (case-insensitive)
        word_counts = Counter(word.lower() for word in words)
        
        # Find max count
        max_count = max(word_counts.values())
        
        # Return first word with max count (preserving original case of first occurrence)
        for word in words:
            if word_counts[word.lower()] == max_count:
                return word.lower()
        
        return None

import pytest



class TestStringProcessor:
    """Test suite for StringProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = StringProcessor()
    
    def test_reverse_words(self):
        """Test reverse_words method."""
        # Basic reversal
        assert self.processor.reverse_words("hello world") == "world hello"
        
        # Multiple spaces between words
        assert self.processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
        
        # Single word
        assert self.processor.reverse_words("single") == "single"
        
        # Empty string
        assert self.processor.reverse_words("") == ""
        
        # Leading and trailing spaces
        assert self.processor.reverse_words("  hello   world  ") == "world hello"
    
    def test_count_vowels(self):
        """Test count_vowels method."""
        # Mixed case
        assert self.processor.count_vowels("hello") == 2  # e, o
        assert self.processor.count_vowels("AEIOU") == 5
        assert self.processor.count_vowels("aeiou") == 5
        
        # No vowels
        assert self.processor.count_vowels("xyz") == 0
        assert self.processor.count_vowels("bcdfg") == 0
        
        # Empty string
        assert self.processor.count_vowels("") == 0
        
        # Mixed with consonants
        assert self.processor.count_vowels("beautiful") == 5  # e, a, u, i, u
    
    def test_is_palindrome(self):
        """Test is_palindrome method."""
        # Classic palindrome with spaces and punctuation
        assert self.processor.is_palindrome("A man a plan a canal Panama") == True
        
        # Simple palindrome
        assert self.processor.is_palindrome("racecar") == True
        assert self.processor.is_palindrome("RaceCar") == True
        
        # Not a palindrome
        assert self.processor.is_palindrome("hello") == False
        assert self.processor.is_palindrome("Python") == False
        
        # Empty string is palindrome
        assert self.processor.is_palindrome("") == True
        
        # Single character
        assert self.processor.is_palindrome("a") == True
        
        # With punctuation
        assert self.processor.is_palindrome("Was it a car or a cat I saw?") == True
    
    def test_caesar_cipher(self):
        """Test caesar_cipher method."""
        # Basic shift
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("ABC", 1) == "BCD"
        
        # With punctuation and spaces (preserved)
        assert self.processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
        
        # Negative shift
        assert self.processor.caesar_cipher("abc", -1) == "zab"
        assert self.processor.caesar_cipher("ABC", -1) == "ZAB"
        
        # Wrap around
        assert self.processor.caesar_cipher("xyz", 1) == "yza"
        assert self.processor.caesar_cipher("XYZ", 1) == "YZA"
        
        # Large shift (equivalent to smaller shift)
        assert self.processor.caesar_cipher("abc", 27) == "bcd"  # 27 % 26 = 1
        
        # Zero shift
        assert self.processor.caesar_cipher("test", 0) == "test"
        
        # Empty string
        assert self.processor.caesar_cipher("", 5) == ""
    
    def test_most_common_word(self):
        """Test most_common_word method."""
        # Classic pangram - "the" appears twice
        assert self.processor.most_common_word(
            "the quick brown fox jumps over the lazy dog"
        ) == "the"
        
        # Clear winner
        assert self.processor.most_common_word("a b a c a") == "a"
        
        # Case insensitivity
        assert self.processor.most_common_word("One two ONE two") == "one"
        
        # Tie - returns first occurrence
        assert self.processor.most_common_word("a b a b") == "a"
        
        # Empty string returns None
        assert self.processor.most_common_word("") is None
        
        # Only whitespace returns None
        assert self.processor.most_common_word("   ") is None
        
        # Single word
        assert self.processor.most_common_word("single") == "single"