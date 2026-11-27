Here's a complete solution split into two files: the class and its pytest tests.

**`string_processor.py`**

```python
"""Utility class for common string operations."""

from typing import Optional


class StringProcessor:
    """Helper methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s.

        Leading/trailing whitespace is removed and internal runs of
        whitespace are collapsed to a single space.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Return True if s is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher with the given shift.

        Only a-z and A-Z are shifted (wrapping around); all other
        characters are left unchanged. Negative shifts are supported.
        """
        result = []
        for c in s:
            if 'a' <= c <= 'z':
                result.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= c <= 'Z':
                result.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(c)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word in s (case-insensitive).

        Ties are broken by first appearance. The original casing of the
        first occurrence is returned. Returns None for empty strings.
        """
        words = s.split()
        if not words:
            return None

        counts = {}
        first_seen = {}
        for i, w in enumerate(words):
            key = w.lower()
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, i)

        # Highest count wins; ties broken by earliest first-appearance.
        best_key = max(counts, key=lambda k: (counts[k], -first_seen[k]))
        return words[first_seen[best_key]]
```

**`test_string_processor.py`**

```python
"""Tests for StringProcessor. Run with: pytest"""

from string_processor import StringProcessor


def test_reverse_words():
    proc = StringProcessor()
    assert proc.reverse_words("hello world foo") == "foo world hello"
    assert proc.reverse_words("  hello   world  ") == "world hello"
    assert proc.reverse_words("a\tb\n c") == "c b a"
    assert proc.reverse_words("") == ""


def test_count_vowels():
    proc = StringProcessor()
    assert proc.count_vowels("Hello World") == 3   # e, o, o
    assert proc.count_vowels("xyz") == 0
    assert proc.count_vowels("AEIOU") == 5


def test_is_palindrome():
    proc = StringProcessor()
    assert proc.is_palindrome("A man, a plan, a canal: Panama") is True
    assert proc.is_palindrome("racecar") is True
    assert proc.is_palindrome("hello") is False


def test_caesar_cipher():
    proc = StringProcessor()
    assert proc.caesar_cipher("abc", 1) == "bcd"
    assert proc.caesar_cipher("abc", -1) == "zab"
    assert proc.caesar_cipher("xyz", 3) == "abc"          # wraps around
    assert proc.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"


def test_most_common_word():
    proc = StringProcessor()
    assert proc.most_common_word("the cat the dog the") == "the"
    assert proc.most_common_word("Apple banana APPLE") == "Apple"  # case-insensitive count
    assert proc.most_common_word("beta alpha beta alpha") == "beta"  # tie: first wins
    assert proc.most_common_word("") is None
```

A few design notes:

- **`reverse_words`** uses `str.split()` (no arguments), which both splits on any whitespace run and strips leading/trailing whitespace — then reversing and rejoining with single spaces gives exactly the requested normalization.
- **`is_palindrome`** keeps alphanumerics and discards everything else, so case, spaces, and punctuation are ignored. (Note: an empty string returns `True`, since the cleaned string is trivially a palindrome.)
- **`caesar_cipher`** relies on Python's `%`, which always returns a non-negative result for a positive modulus — so negative shifts wrap correctly without special-casing.
- **`most_common_word`** counts case-insensitively but returns the original casing of the winning word's first occurrence. Tie-breaking is handled via the first-appearance index rather than relying on dict ordering, so it's robust.

To run the tests:

```bash
pytest test_string_processor.py
```