Here's the `StringProcessor` class along with 5 pytest tests covering each method:

```python
from typing import Optional
from collections import Counter


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.

        Leading/trailing whitespace is removed and multiple internal
        spaces are collapsed into a single space.
        """
        return ' '.join(reversed(s.split()))

    def count_vowels(self, s: str) -> int:
        """Count the vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for ch in s.lower() if ch in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher with the given shift.

        Only shifts a-z and A-Z (wrapping around); all other characters
        are left unchanged. Negative shifts are supported.
        """
        def shift_char(ch: str) -> str:
            if 'a' <= ch <= 'z':
                return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
            if 'A' <= ch <= 'Z':
                return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
            return ch
        return ''.join(shift_char(ch) for ch in s)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).

        Ties are broken by order of first appearance. The result is
        returned in lowercase. Returns None for empty strings.
        """
        words = s.split()
        if not words:
            return None
        counts = Counter(w.lower() for w in words)
        # Iterate in first-appearance order so ties resolve to the earliest word
        for w in dict.fromkeys(w.lower() for w in words):
            if counts[w] == max(counts.values()):
                return w
        return None
```

```python
import pytest
from StringProcessor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("one two three") == "three two one"
    assert sp.reverse_words("   ") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3  # e, o, o
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("rhythm") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("Racecar") is True
    assert sp.is_palindrome("hello") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"           # negative shift
    assert sp.caesar_cipher("xyz", 3) == "abc"             # wraps around
    assert sp.caesar_cipher("Hello 123!", 2) == "Jgnnq 123!"  # non-letters unchanged


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("Banana apple cherry") == "banana"  # tie → first
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

A few design notes:

- **`reverse_words`**: `str.split()` (no arguments) handles whitespace normalization and stripping in one step, then `reversed()` reverses word order.
- **`is_palindrome`**: strips everything that isn't alphanumeric (which covers spaces, punctuation, and any emoji/symbols), lowercases, then compares against its reverse.
- **`caesar_cipher`**: uses modulo arithmetic on the alphabet range, so negative shifts and wrap-around work naturally. `shift` is any int, including values ≥ 26 or ≤ -26.
- **`most_common_word`**: `dict.fromkeys` preserves first-appearance order while deduplicating, so the tie-breaking logic picks the earliest word with the max count. The result is returned lowercased, which is why `"Banana"` in the tie test returns `"banana"` — happy to switch it to preserve original casing if you prefer that.

To run the tests: `pytest -v` (or `python -m pytest -v`) from the directory containing the code.