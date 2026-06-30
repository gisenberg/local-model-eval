import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to the string, shifting only a-z/A-Z characters. Supports negative shifts."""
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive). Returns the first one if tied."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        # Iterate in original order to return the first word in case of a tie
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("z", -1) == "y"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie: returns first encountered
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None