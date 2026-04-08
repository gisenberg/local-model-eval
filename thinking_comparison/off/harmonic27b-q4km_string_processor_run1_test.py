from typing import Optional
import re

class StringProcessor:
    """A class providing various string processing utilities."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        Multiple spaces become single space, leading/trailing spaces removed.
        """
        if not s.strip():
            return ""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if string is a palindrome, ignoring case, spaces, and punctuation.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift. Only shifts a-z and A-Z.
        Supports negative shifts. Other characters remain unchanged.
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
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first. Return None for empty strings.
        """
        if not s.strip():
            return None
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        # Count frequencies while preserving order of first appearance
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Find the word with maximum frequency (first occurrence in case of tie)
        max_count = max(word_count.values())
        for word in words:
            if word_count[word] == max_count:
                return word
        return None

import pytest
from typing import Optional
  # Replace with actual module name

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("   ") == ""

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("hello") == 2  # e, o
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("bcdfg") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("rhythm") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("xyz", 2) == "zab"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("hello world hello") == "hello"
    assert processor.most_common_word("The quick brown fox jumps over the lazy dog") == "the"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None
    # Test tie-breaking (first occurrence wins)
    assert processor.most_common_word("apple banana apple banana") == "apple"