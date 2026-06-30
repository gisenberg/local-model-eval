import re
from typing import Optional
from collections import Counter


class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the string."""
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
        """Apply a Caesar cipher shift to alphabetic characters only. Supports negative shifts."""
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
        Returns the first occurrence in case of a tie. Returns None if no words."""
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
        
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Return the first word in original order that matches the max count
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("the quick brown fox") == "fox brown quick the"
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3  # e, o, o
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("ABC", -1) == "ZAB"
    assert sp.caesar_cipher("a!b@c", 1) == "b!c@d"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat") == "cat"
    assert sp.most_common_word("a b c") == "a"  # Tie: first occurrence wins
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None