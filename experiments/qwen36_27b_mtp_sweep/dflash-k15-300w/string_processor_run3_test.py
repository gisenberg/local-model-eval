import re
from typing import Optional
from collections import Counter

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in a string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher to a string, shifting only a-z/A-Z characters.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Returns the most common word in a string (case-insensitive).
        
        If multiple words are tied for the highest frequency, returns the first 
        one encountered in the original string.
        
        Args:
            s: Input string to analyze.
            
        Returns:
            The most frequent word, or None if the string contains no words.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Preserve original order to handle ties correctly
        for word in words:
            if counts[word] == max_count:
                return word
        return None

import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("") == ""

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU") == 5

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert StringProcessor.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tied, returns first
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None