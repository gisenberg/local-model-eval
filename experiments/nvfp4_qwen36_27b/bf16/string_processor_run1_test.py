import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in the given string.
        Multiple consecutive spaces are normalized to single spaces."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies a Caesar cipher to the string, shifting only a-z/A-Z characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Returns the most frequent word in the string (case-insensitive).
        Returns the first encountered word in case of a tie. Returns None if no words."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts = {}
        max_count = 0
        most_common = None
        
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
                
        return most_common


# ========================
# Pytest Test Suite
# ========================
import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert processor.reverse_words("single") == "single"

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("rhythm") == 0
    assert processor.count_vowels("AEIOU") == 5

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert processor.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert processor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word(processor):
    assert processor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert processor.most_common_word("cat dog cat bird dog") == "cat"  # Tie: 'cat' appears first
    assert processor.most_common_word("") is None