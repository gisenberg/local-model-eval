Here's the complete implementation with type hints, docstrings, and 5 pytest tests (one per method):

```python
import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string. Multiple spaces become single spaces, leading/trailing spaces removed."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for char in s if char.lower() in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift. Only shifts a-z and A-Z, leaves others unchanged. Supports negative shifts."""
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive). Ties return the first occurrence. Returns None for empty strings."""
        if not s.strip():
            return None
        
        # Extract words, ignoring punctuation
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        # Python 3.7+ dicts preserve insertion order, so the first word with max_count wins ties
        for word, count in counts.items():
            if count == max_count:
                return word
        return None
```

### Pytest Tests

```python
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is fun") == "fun is Python"
    assert sp.reverse_words("") == ""
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("hello") == 2
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") == True
    assert sp.is_palindrome("racecar") == True
    assert sp.is_palindrome("hello") == False
    assert sp.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("ABC", 1) == "BCD"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("abc", -1) == "zab"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("apple banana apple") == "apple"
    assert sp.most_common_word("one two three") == "one"  # Tie: first occurrence wins
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

### Key Implementation Details:
1. **`reverse_words`**: Uses `str.split()` without arguments, which automatically handles multiple spaces and strips leading/trailing whitespace.
2. **`caesar_cipher`**: Leverages Python's modulo operator, which correctly handles negative shifts (e.g., `-1 % 26 == 25`).
3. **`most_common_word`**: Uses `re.findall(r'\b\w+\b', ...)` to cleanly extract words while ignoring punctuation. Python 3.7+ guarantees dictionary insertion order, making tie-breaking by first occurrence straightforward.
4. **Type Hints & Docstrings**: All methods include proper type annotations and concise docstrings as requested.

Run the tests with: `pytest test_file.py -v`