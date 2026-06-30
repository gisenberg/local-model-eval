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
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters only. Supports negative shifts."""
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        Returns the first word encountered if there's a tie."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        # Dictionary preserves insertion order in Python 3.7+, guaranteeing first-occurrence tie-breaking
        for word in counts:
            if counts[word] == max_count:
                return word

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
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World! 123", 13) == "Uryyb, Jbeyq! 123"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"  # Tie: first occurrence wins
    assert StringProcessor.most_common_word("") is None