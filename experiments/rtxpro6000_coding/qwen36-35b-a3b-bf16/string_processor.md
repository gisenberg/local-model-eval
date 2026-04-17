Here's the complete implementation with type hints, docstrings, and 5 pytest tests:

```python
import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string, collapsing multiple spaces."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s if c.lower() in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters only."""
        shift = shift % 26
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Ties broken by first occurrence."""
        words = s.lower().split()
        if not words:
            return None

        counts = {}
        first_index = {}
        for i, word in enumerate(words):
            counts[word] = counts.get(word, 0) + 1
            if word not in first_index:
                first_index[word] = i

        max_count = max(counts.values())
        return min((w for w, c in counts.items() if c == max_count), key=lambda w: first_index[w])
```

### Pytest Tests

```python
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("Python is awesome") == "awesome is Python"
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
    assert sp.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("abc", -1) == "zab"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("apple banana apple") == "apple"
    assert sp.most_common_word("one two three") == "one"  # tie, first wins
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

### Key Implementation Details:
- **`reverse_words`**: Uses `str.split()` which automatically handles multiple spaces and strips leading/trailing whitespace.
- **`is_palindrome`**: Regex `[^a-zA-Z0-9]` efficiently strips all non-alphanumeric characters before comparison.
- **`caesar_cipher`**: Python's `%` operator correctly handles negative shifts (e.g., `-1 % 26 == 25`), making wrap-around logic clean.
- **`most_common_word`**: Tracks both frequency and first occurrence index to guarantee correct tie-breaking behavior without relying on dictionary insertion order assumptions.
- Run tests with: `pytest your_file.py -v`