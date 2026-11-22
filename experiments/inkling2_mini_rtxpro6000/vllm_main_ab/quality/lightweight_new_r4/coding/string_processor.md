Here's the `StringProcessor` class and the tests:

```python
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words (not characters).
        Collapses multiple spaces to one and removes leading/trailing spaces."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u), case-insensitive."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if s is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(c for c in s.lower() if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift a-z and A-Z letters by `shift` positions (wraps around).
        Non-letter characters are left unchanged. Supports negative shifts."""
        shift = shift % 26
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
        """Return the most frequently occurring word (case-insensitive).
        On ties, returns the word that appears first. Returns None for empty strings."""
        words = s.split()
        if not words:
            return None
        counts = {}
        for w in words:
            key = w.lower()
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values())
        for w in words:  # first occurrence among the tied/max words
            if counts[w.lower()] == max_count:
                return w
        return None  # unreachable
```

```python
# test_string_processor.py
from StringProcessor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3  # e, o, o
    assert sp.count_vowels("AEIOU") == 5


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("abc", 27) == "bcd"  # shift wraps beyond 26


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("cat dog") == "cat"   # tie -> first occurrence
    assert sp.most_common_word("   ") is None
```

Notes on design decisions:

- **`is_palindrome`**: keeps alphanumeric characters and drops everything else, so digits are retained (e.g., "A1" is a palindrome). If you'd rather strip digits too, change `c.isalnum()` to `c.isalpha()`.
- **`caesar_cipher`**: `shift % 26` normalizes the shift, so negative values (and shifts ≥ 26) work correctly without special-casing.
- **`most_common_word`**: returns the word in its original capitalization (the first occurrence of the winning key), while frequency matching is case-insensitive.

Run the tests with:

```bash
pytest test_string_processor.py -v
```