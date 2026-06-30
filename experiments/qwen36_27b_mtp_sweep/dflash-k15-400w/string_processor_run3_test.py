import re
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in the given string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order, separated by single spaces.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher to the string, shifting only alphabetic characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (supports negative values).
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Returns the most common word in the string (case-insensitive).
        
        In case of a tie, returns the word that appeared first in the original string.
        Returns None if the string contains no words.
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word, or None.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = 0
        most_common: Optional[str] = None
        for word, count in counts.items():
            if count > max_count:
                max_count = count
                most_common = word
                
        return most_common


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tie: first occurrence wins
    assert sp.most_common_word("... ,,, !!!") is None