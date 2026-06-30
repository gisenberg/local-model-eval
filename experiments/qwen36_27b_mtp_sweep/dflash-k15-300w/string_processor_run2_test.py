from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case and non-alphabetic characters."""
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
        
        Returns the first occurrence if there's a tie. Returns None for empty strings.
        """
        words = s.lower().split()
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

@pytest.fixture
def sp():
    return StringProcessor()

def test_reverse_words(sp):
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels(sp):
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome(sp):
    assert sp.is_palindrome("A man, a plan, a canal: Panama")
    assert sp.is_palindrome("racecar")
    assert not sp.is_palindrome("hello")

def test_caesar_cipher(sp):
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word(sp):
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # first if tied
    assert sp.most_common_word("") is None