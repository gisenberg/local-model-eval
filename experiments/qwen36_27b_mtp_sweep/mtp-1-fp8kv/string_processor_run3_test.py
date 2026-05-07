import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string.
        
        Note: Normalizes multiple consecutive whitespace characters into single spaces.
        
        Args:
            s: Input string containing words.
            
        Returns:
            String with words in reversed order.
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
            s: Input string.
            shift: Number of positions to shift (supports negative values).
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most common word in a string (case-insensitive).
        
        If multiple words are tied for the highest frequency, returns the first one 
        encountered in the original text.
        
        Args:
            s: Input string.
            
        Returns:
            The most common word, or None if the string contains no words.
        """
        if not s.strip():
            return None
            
        # Extract words, ignoring punctuation and converting to lowercase
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None
            
        # Count frequencies while preserving insertion order (Python 3.7+)
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        for word, count in counts.items():
            if count == max_count:
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
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"  # tied, returns first
    assert StringProcessor.most_common_word("no words here!") == "no"         # all tied, returns first
    assert StringProcessor.most_common_word("") is None