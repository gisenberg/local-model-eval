import re
from collections import Counter
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels (a, e, i, o, u) case-insensitively."""
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher shift to alphabetic characters."""
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Calculate shift using modulo 26 to handle negative shifts and overflows
                shifted = chr((ord(char) - start + shift) % 26 + start)
                result.append(shifted)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Ties go to the first occurrence."""
        if not s or not s.strip():
            return None
        
        # Normalize to lowercase and split into words
        words = s.lower().split()
        counts = Counter(words)
        
        # Find the max frequency
        max_freq = max(counts.values())
        
        # Return the first word in the original list that matches the max frequency
        for word in words:
            if counts[word] == max_freq:
                return word
        return None

# --- Pytest Tests ---
# To run these, save as test_string_processor.py and run `pytest` in terminal.

import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("  hello world  ") == "world hello"
    assert processor.reverse_words("The quick brown fox") == "fox brown quick The"
    assert processor.reverse_words("multiple   spaces") == "spaces multiple"

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("Pythn") == 0
    assert processor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("Hello World!", 3) == "Khoor Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("xyz", 1) == "yza"

def test_most_common_word(processor):
    assert processor.most_common_word("Apple banana apple orange") == "apple"
    assert processor.most_common_word("dog cat dog cat") == "dog"  # Tie: dog appears first
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None