from typing import Optional
import re

class StringProcessor:
    """A class providing various string processing utilities."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        """
        # Split by whitespace (handles multiple spaces), then reverse and join
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                offset = ord('a')
                shifted = (ord(char) - offset + shift) % 26
                result.append(chr(offset + shifted))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                offset = ord('A')
                shifted = (ord(char) - offset + shift) % 26
                result.append(chr(offset + shifted))
            else:
                # Leave non-alphabetic characters unchanged
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.
        Return None for empty strings.
        """
        if not s or not s.strip():
            return None
            
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\w+', s.lower())
        
        if not words:
            return None
            
        # Count frequencies while preserving order of first appearance
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
            
        # Find the word with maximum frequency (first occurrence wins ties)
        max_count = max(word_count.values())
        for word in words:
            if word_count[word] == max_count:
                return word
        return None

import pytest
from typing import Optional
  # Replace with actual module name

class TestStringProcessor:
    def setup_method(self):
        self.processor = StringProcessor()

    def test_reverse_words(self):
        """Test reverse_words method"""
        assert self.processor.reverse_words("hello world") == "world hello"
        assert self.processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
        assert self.processor.reverse_words("single") == "single"
        assert self.processor.reverse_words("") == ""

    def test_count_vowels(self):
        """Test count_vowels method"""
        assert self.processor.count_vowels("hello") == 2  # e, o
        assert self.processor.count_vowels("AEIOU") == 5
        assert self.processor.count_vowels("bcdfg") == 0
        assert self.processor.count_vowels("") == 0

    def test_is_palindrome(self):
        """Test is_palindrome method"""
        assert self.processor.is_palindrome("racecar") is True
        assert self.processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert self.processor.is_palindrome("hello") is False
        assert self.processor.is_palindrome("Was it a car or a cat I saw?") is True

    def test_caesar_cipher(self):
        """Test caesar_cipher method"""
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("ABC", 1) == "BCD"
        assert self.processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
        assert self.processor.caesar_cipher("abc", -1) == "zab"
        assert self.processor.caesar_cipher("xyz", 1) == "yza"

    def test_most_common_word(self):
        """Test most_common_word method"""
        assert self.processor.most_common_word("hello world hello") == "hello"
        assert self.processor.most_common_word("a b a c b a") == "a"
        assert self.processor.most_common_word("tie tie tie") == "tie"
        assert self.processor.most_common_word("") is None
        assert self.processor.most_common_word("   ") is None
        # Test case-insensitivity and tie-breaking
        assert self.processor.most_common_word("Hello hello HELLO") == "hello"