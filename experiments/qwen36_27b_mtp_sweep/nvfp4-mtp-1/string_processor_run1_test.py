import re
from typing import Optional
from collections import Counter


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in the given string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies a Caesar cipher to the string, shifting only a-z/A-Z characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most common word in the string (case-insensitive).
        Returns the first word encountered if there's a tie. Returns None for empty strings."""
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
        counts = Counter(words)
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("ABC", -1) == "ZAB"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("123 @#$", 5) == "123 @#$"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat") == "cat"
    assert sp.most_common_word("a b c") == "a"  # Tie scenario: first encountered wins
    assert sp.most_common_word("") is None
    assert sp.most_common_word("!!! ???") is None