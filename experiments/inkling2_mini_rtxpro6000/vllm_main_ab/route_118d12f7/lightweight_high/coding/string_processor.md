Here's the implementation and tests. I'll split them into two files, which is the usual pytest convention.

**`string_processor.py`**

```python
from collections import Counter
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words (not characters). Collapses runs of
        whitespace into a single space and strips leading/trailing spaces."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s.lower() if c in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Return True if the string reads the same forwards and backwards,
        ignoring case, spaces, and punctuation."""
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher: shift each a-z/A-Z letter by `shift`
        positions (wraps around). Non-alphabetic characters are unchanged.
        Negative shifts are supported."""
        result = []
        for c in s:
            if "a" <= c <= "z":
                result.append(chr((ord(c) - ord("a") + shift) % 26 + ord("a")))
            elif "A" <= c <= "Z":
                result.append(chr((ord(c) - ord("A") + shift) % 26 + ord("A")))
            else:
                result.append(c)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        On a tie, return the one that appears first (in original casing).
        Returns None for empty/whitespace-only strings."""
        words = s.split()
        if not words:
            return None
        counts = Counter(w.lower() for w in words)
        max_count = max(counts.values())
        for w in words:
            if counts[w.lower()] == max_count:
                return w
        return None
```

**`test_string_processor.py`**

```python
import pytest

from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("  hello   world  foo ") == "foo world hello"
    assert processor.reverse_words("") == ""


def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU") == 5


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("Not a palindrome") is False


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("Hello", -2) == "Fggnoc"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"  # wraps around


def test_most_common_word(processor):
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("b a b") == "b"  # tie -> first occurrence
    assert processor.most_common_word("   ") is None
```

Run with:

```bash
pytest
```

A few design notes:

- **`is_palindrome`** strips everything that isn't alphanumeric (spaces, punctuation), lowercases, then compares with its reverse. Digits are kept since they aren't punctuation.
- **`caesar_cipher`** uses Python's modulo, so negative shifts wrap correctly in either direction.
- **`most_common_word`** counts case-insensitively but returns the first-occurring original-cased spelling (e.g. `"hello HELLO hello"` → `"hello"`). The tie rule falls out naturally because the loop returns the first word whose count matches the maximum.

The `isalnum()` choice means characters like accented letters (e.g. `é`) are treated as letters. If you want strictly ASCII, swap the filter for `c.isascii() and c.isalnum()`.