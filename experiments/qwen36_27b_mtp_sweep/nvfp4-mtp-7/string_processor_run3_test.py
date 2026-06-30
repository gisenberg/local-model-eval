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
        return sum(1 for c in s.lower() if c in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to a string, shifting only a-z/A-Z characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged.
        """
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
        """Find the most common word in a string (case-insensitive).
        Returns the first word by appearance if there's a tie. Returns None if no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        return next(word for word in words if counts[word] == max_count)

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
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("Was it a car I saw?") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("xyz", 1) == "yza"
    assert StringProcessor.caesar_cipher("ABC", -1) == "ZAB"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie broken by first appearance
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("  !@#  ") is None