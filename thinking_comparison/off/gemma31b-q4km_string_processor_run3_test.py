import re
from collections import Counter
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome, ignoring case, spaces, and punctuation."""
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, supporting negative shifts."""
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Formula: (current_pos + shift) % 26
                shifted = chr(start + (ord(char) - start + shift) % 26)
                result.append(shifted)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Ties go to the first occurrence."""
        if not s.strip():
            return None
        
        # Normalize to lowercase and split by non-alphanumeric characters
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        # max() in Python is stable, so it returns the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])

# --- Pytest Tests ---
import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("  hello world  ") == "world hello"
    assert processor.reverse_words("Python is   awesome") == "awesome is Python"

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("xyz") == 0

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("Hello World!", 3) == "Khoor Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"

def test_most_common_word(processor):
    assert processor.most_common_word("Apple banana apple orange") == "apple"
    assert processor.most_common_word("Dog cat dog cat") == "dog"  # Tie: dog comes first
    assert processor.most_common_word("") is None