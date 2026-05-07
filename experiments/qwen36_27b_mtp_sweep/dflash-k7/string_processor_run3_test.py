import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive)."""
        return sum(1 for char in s if char.lower() in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to a string, shifting only a-z/A-Z characters.
        Supports negative shifts and wraps around the alphabet."""
        result = []
        shift = shift % 26  # Python's modulo handles negatives correctly
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
        """Return the most common word in a string (case-insensitive).
        Returns the first word encountered if there's a tie.
        Returns None if the string contains no words."""
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        # Iterate in original order to guarantee "first if tied" behavior
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
    assert StringProcessor.is_palindrome("") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("xyz", -1) == "wxy"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple cherry") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie: first wins
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("   ") is None