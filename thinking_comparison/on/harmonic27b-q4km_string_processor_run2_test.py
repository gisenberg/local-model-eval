from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        """
        if not s or not s.strip():
            return ""
        
        # Split by whitespace (handles multiple spaces), reverse, join with single space
        words = s.split()
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        vowels = 'aeiouAEIOU'
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        # Filter to only alphanumeric characters and convert to lowercase
        filtered = ''.join(char.lower() for char in s if char.isalnum())
        return filtered == filtered[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
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
                # Keep non-alphabetic characters unchanged
                result.append(char)
        
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        
        If tied, return the one that appears first.
        Return None for empty strings.
        """
        if not s or not s.strip():
            return None
        
        # Split into words and convert to lowercase
        words = s.split()
        if not words:
            return None
        
        # Count frequencies while preserving order
        word_counts = Counter()
        first_occurrence = {}
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            word_counts[word_lower] += 1
            if word_lower not in first_occurrence:
                first_occurrence[word_lower] = i
        
        if not word_counts:
            return None
        
        # Find maximum frequency
        max_freq = max(word_counts.values())
        
        # Get all words with maximum frequency
        most_common = [word for word, count in word_counts.items() if count == max_freq]
        
        # Return the one that appears first
        return min(most_common, key=lambda w: first_occurrence[w])


# Pytest tests
import pytest

class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("single") == "single"
        assert processor.reverse_words("") == ""
    
    def test_count_vowels(self):
        processor = StringProcessor()
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("bcdfg") == 0
        assert processor.count_vowels("") == 0
    
    def test_is_palindrome(self):
        processor = StringProcessor()
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        assert processor.is_palindrome("racecar") == True
        assert processor.is_palindrome("hello") == False
        assert processor.is_palindrome("") == True
    
    def test_caesar_cipher(self):
        processor = StringProcessor()
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("ABC", 1) == "BCD"
        assert processor.caesar_cipher("abc", -1) == "zab"
        assert processor.caesar_cipher("hello!", 3) == "khoor!"
    
    def test_most_common_word(self):
        processor = StringProcessor()
        assert processor.most_common_word("hello world hello") == "hello"
        assert processor.most_common_word("a b a b") == "a"
        assert processor.most_common_word("") is None
        assert processor.most_common_word("   ") is None