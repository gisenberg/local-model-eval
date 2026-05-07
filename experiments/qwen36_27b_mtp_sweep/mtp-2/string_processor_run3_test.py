import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in the string. Multiple/leading/trailing spaces are normalized."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher to the string, shifting only a-z/A-Z characters.
        Supports positive and negative shifts. Non-alphabetic characters remain unchanged."""
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
        """Returns the most common word in the string (case-insensitive).
        If multiple words share the highest frequency, returns the first one encountered."""
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word

import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("rhythm") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"  # Tie: 'cat' appears first
    assert StringProcessor.most_common_word("") is None