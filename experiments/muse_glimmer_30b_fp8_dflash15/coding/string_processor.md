```python
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Return words in reverse order, single spaces, trimmed."""
        return " ".join(reversed(s.strip().split()))

    def count_vowels(self, s: str) -> int:
        """Count a e i o u case-insensitive."""
        vowels = set("aeiouAEIOU")
        return sum(1 for ch in s if ch in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Palindrome check ignoring case, spaces and punctuation."""
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift only A-Z / a-z, other chars unchanged. Supports negative shift."""
        res = []
        for ch in s:
            if 'a' <= ch <= 'z':
                base = ord('a')
                res.append(chr((ord(ch) - base + shift) % 26 + base))
            elif 'A' <= ch <= 'Z':
                base = ord('A')
                res.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                res.append(ch)
        return "".join(res)

    def most_common_word(self, s: str) -> Optional[str]:
        """Most frequent word case-insensitive, first on tie. None if empty."""
        if not s or not s.strip():
            return None
        words = s.split()
        counts = {}
        first_idx = {}
        for i, w in enumerate(words):
            key = w.lower()
            if key not in counts:
                counts[key] = 0
                first_idx[key] = i
            counts[key] += 1

        best_key = None
        best_count = -1
        best_first = float('inf')
        for key, cnt in counts.items():
            idx = first_idx[key]
            if cnt > best_count or (cnt == best_count and idx < best_first):
                best_count = cnt
                best_first = idx
                best_key = key
        return best_key
```

```python
# test_string_processor.py
import pytest
from string_processor import StringProcessor

def test_reverse_words():
    p = StringProcessor()
    assert p.reverse_words("  hello   world  ") == "world hello"
    assert p.reverse_words("a  b   c") == "c b a"
    assert p.reverse_words("") == ""

def test_count_vowels():
    p = StringProcessor()
    assert p.count_vowels("Hello World") == 3
    assert p.count_vowels("AEIOUaeiou") == 10
    assert p.count_vowels("bcdfg") == 0

def test_is_palindrome():
    p = StringProcessor()
    assert p.is_palindrome("A man, a plan, a canal: Panama") is True
    assert p.is_palindrome("No lemon, no melon") is True
    assert p.is_palindrome("hello") is False

def test_caesar_cipher():
    p = StringProcessor()
    assert p.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert p.caesar_cipher("def", -3) == "abc"
    assert p.caesar_cipher("Hello, World!", 0) == "Hello, World!"
    assert p.caesar_cipher("xyz", 2) == "zab"

def test_most_common_word():
    p = StringProcessor()
    assert p.most_common_word("Hello hello HELLO world") == "hello"
    assert p.most_common_word("") is None
    assert p.most_common_word("apple banana apple banana cherry") == "apple"
```

Run with `pytest test_string_processor.py`.