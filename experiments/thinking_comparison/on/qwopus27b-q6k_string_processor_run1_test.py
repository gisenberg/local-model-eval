from typing import Optional
import re


class StringProcessor:
    """A class for various string processing operations."""
    
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
            Count of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if string is a palindrome, ignoring case, spaces, and punctuation.
        
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
        
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)
            
        Returns:
            Encrypted string
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        
        If tied, return the one that appears first.
        Return None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if empty
        """
        if not s or not s.strip():
            return None
        
        words = s.lower().split()
        counts = {}
        
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        
        max_count = max(counts.values())
        
        for word in words:
            if counts[word] == max_count:
                return word
        
        return None


# Pytest tests
import pytest


class TestStringProcessor:
    def setup_method(self):
        self.processor = StringProcessor()
    
    def test_reverse_words(self):
        """Test reverse_words method."""
        assert self.processor.reverse_words("hello world") == "world hello"
        assert self.processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
        assert self.processor.reverse_words("single") == "single"
        assert self.processor.reverse_words("") == ""
    
    def test_count_vowels(self):
        """Test count_vowels method."""
        assert self.processor.count_vowels("hello") == 2  # e, o
        assert self.processor.count_vowels("AEIOU") == 5
        assert self.processor.count_vowels("bcdfg") == 0
        assert self.processor.count_vowels("") == 0
    
    def test_is_palindrome(self):
        """Test is_palindrome method."""
        assert self.processor.is_palindrome("racecar") is True
        assert self.processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert self.processor.is_palindrome("hello") is False
        assert self.processor.is_palindrome("") is True
    
    def test_caesar_cipher(self):
        """Test caesar_cipher method."""
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("abc", -1) == "zab"
        assert self.processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
        assert self.processor.caesar_cipher("xyz", 3) == "abc"
    
    def test_most_common_word(self):
        """Test most_common_word method."""
        assert self.processor.most_common_word("hello world hello") == "hello"
        assert self.processor.most_common_word("a b a c c") == "a"  # tied, first appears first
        assert self.processor.most_common_word("") is None
        assert self.processor.most_common_word("   ") is None
        assert self.processor.most_common_word("Hello hello HELLO") == "hello"