```python
import re
from typing import Optional
from collections import Counter

class StringProcessor:
    """A utility class for advanced string manipulation operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string. 
        Multiple spaces are collapsed to single spaces, and leading/trailing spaces are removed.
        """
        # split() without arguments splits by any whitespace and discards empty strings
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift. Only shift a-z and A-Z, 
        leave other characters unchanged. Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift within lowercase range
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                # Shift within uppercase range
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                # Keep non-alphabetic characters as is
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive). 
        If tied, return the one that appears first. Return None for empty strings.
        """
        words = s.lower().split()
        if not words:
            return None
        
        counts = Counter(words)
        max_freq = max(counts.values())
        
        # Iterate through original words to maintain first-appearance order for ties
        for word in words:
            if counts[word] == max_freq:
                return word
        return None

# --- Pytest Tests ---

import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("  Hello   World  ") == "World Hello"
    assert processor.reverse_words("Python is awesome") == "awesome is Python"
    assert processor.reverse_words("") == ""

def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU aeiou") == 10
    assert processor.count_vowels("bcdfg") == 0

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False
    assert processor.is_palindrome(" ") is True  # Empty after cleaning is technically palindrome

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("Hello World!", 3) == "Khoor Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"
    assert processor.caesar_cipher("123 !@#", 5) == "123 !@#"

def test_most_common_word(processor):
    assert processor.most_common_word("The cat sat on the mat") == "the"
    assert processor.most_common_word("Apple banana apple banana cherry") == "apple"  # Tie: apple comes first
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None
```

### Key Implementation Details:

1.  **`reverse_words`**: Uses `s.split()` which automatically handles multiple spaces and trims the string, then reverses the resulting list.
2.  **`count_vowels`**: Uses a generator expression with `sum()` for memory efficiency.
3.  **`is_palindrome`**: Uses `isalnum()` to filter out punctuation and spaces, then compares the string to its slice reverse `[::-1]`.
4.  **`caesar_cipher`**: Uses modulo 26 arithmetic to ensure the shift wraps around the alphabet correctly, handling both positive and negative integers.
5.  **`most_common_word`**: Uses `collections.Counter` for frequency counting, but iterates through the original word list to ensure that in the event of a tie, the word that appeared first in the string is returned.

### How to run the tests:
Save the code to a file (e.g., `string_processor.py`) and run:
```bash
pytest string_processor.py
```