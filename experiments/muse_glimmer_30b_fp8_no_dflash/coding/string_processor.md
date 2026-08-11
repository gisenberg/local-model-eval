```python
from typing import Optional


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Return words in reverse order, single spaced, trimmed."""
        words = s.strip().split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count a,e,i,o,u case-insensitive."""
        return sum(1 for c in s.lower() if c in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Palindrome check ignoring case, spaces and punctuation."""
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift a-z / A-Z, leave other chars unchanged. Negative shifts supported."""
        shift %= 26
        out = []
        for c in s:
            if "a" <= c <= "z":
                base = ord("a")
                out.append(chr((ord(c) - base + shift) % 26 + base))
            elif "A" <= c <= "Z":
                base = ord("A")
                out.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                out.append(c)
        return "".join(out)

    def most_common_word(self, s: str) -> Optional[str]:
        """Most frequent word case-insensitive. First appearance wins ties. None if empty."""
        if not s or not s.strip():
            return None
        words = s.split()
        counts = {}
        first_seen = {}
        for w in words:
            lw = w.lower()
            if lw not in counts:
                counts[lw] = 0
                first_seen[lw] = w
            counts[lw] += 1

        best_lw = None
        best_cnt = -1
        for lw, cnt in counts.items():
            if cnt > best_cnt:
                best_cnt = cnt
                best_lw = lw
        return first_seen[best_lw]
```

**Tests**

```python
import pytest
from string_processor import StringProcessor

sp = StringProcessor()

def test_reverse_words():
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("one") == "one"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    assert sp.count_vowels("AeIoU") == 5
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome():
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("Racecar") is True
    assert sp.is_palindrome("Python") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    assert sp.caesar_cipher("abc XYZ", 2) == "cde ZAB"
    assert sp.caesar_cipher("abc XYZ", -1) == "zab WXY"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("xyz", 52) == "xyz"  # full rotations

def test_most_common_word():
    assert sp.most_common_word("Hello hello HELLO world") == "Hello"
    assert sp.most_common_word("apple banana apple banana") == "apple"  # tie -> first
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

Run with `pytest test_stringprocessor.py`.