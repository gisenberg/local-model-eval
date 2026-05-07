import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            A string with the words in reverse order.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            The total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply a Caesar cipher to a string, shifting only alphabetic characters.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift (supports negative values).
            
        Returns:
            The transformed string with only a-z/A-Z characters shifted.
        """
        result = []
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
            s: Input string to analyze.
            
        Returns:
            The most frequent word. If multiple words tie, returns the first one 
            encountered in the original text. Returns None if no words are found.
        """
        if not s or not s.strip():
            return None
            
        # Extract words, ignoring punctuation and converting to lowercase
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts = {}
        order = []
        for word in words:
            if word not in counts:
                counts[word] = 0
                order.append(word)
            counts[word] += 1
            
        max_count = 0
        most_common = None
        for word in order:
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
        return most_common


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
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("bcdfg") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello world") is False

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    # Tie-breaking: 'cat' appears first in the text
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"
    assert StringProcessor.most_common_word("") is None