import re
from typing import Optional

class StringProcessor:
    """Utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string.
        Multiple consecutive whitespace characters are treated as a single delimiter.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to the string with the given shift.
        Only alphabetic characters (a-z, A-Z) are shifted. Other characters remain unchanged.
        Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted)
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        If multiple words are tied for the highest frequency, returns the first one encountered.
        Returns None if the string contains no words.
        """
        if not s:
            return None
            
        # Extract alphabetic words only
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None

        counts = {}
        max_count = 0
        most_common = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            # Strict greater-than ensures we keep the first word in case of ties
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
                
        return most_common


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3  # e, o, o
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") == True
    assert sp.is_palindrome("racecar") == True
    assert sp.is_palindrome("hello") == False
    assert sp.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("abc XYZ", -1) == "zab WXY"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("a", 26) == "a"  # Full rotation

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tie: returns first encountered
    assert sp.most_common_word("The THE the") == "the"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None