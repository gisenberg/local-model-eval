Here's the implementation and the tests:

**`string_processor.py`**

```python
from typing import Optional


class StringProcessor:
    """Utility class providing string manipulation methods."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s. Collapse multiple spaces into
        a single space and strip leading/trailing whitespace."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for ch in s if ch.lower() in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Return True if s is a palindrome, ignoring case, spaces, and
        punctuation (alphanumeric characters only, lowercased)."""
        cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift letters a-z / A-Z by shift positions (negative shifts allowed).
        All other characters are left unchanged."""
        shift %= 26
        result = []
        for ch in s:
            if 'a' <= ch <= 'z':
                result.append(chr((ord(ch) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= ch <= 'Z':
                result.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(ch)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        On ties, return the word that appears first. Preserves the original
        casing of the word's first occurrence. Returns None for empty strings."""
        words = s.split()
        if not words:
            return None
        counts = {}
        for word in words:
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values())
        for word in words:
            if counts[word.lower()] == max_count:
                return word
        return None  # unreachable, but satisfies the type checker
```

**`test_string_processor.py`**

```python
import pytest

from string_processor import StringProcessor

processor = StringProcessor()


def test_reverse_words():
    assert processor.reverse_words("  hello   world  ") == "world hello"
    assert processor.reverse_words("one") == "one"
    assert processor.reverse_words("") == ""


def test_count_vowels():
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("AEIOU") == 5


def test_is_palindrome():
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False


def test_caesar_cipher():
    assert processor.caesar_cipher("Hello", 3) == "Khoor"
    assert processor.caesar_cipher("Hello", -3) == "Ebiil"
    assert processor.caesar_cipher("abc!", 1) == "bcd!"


def test_most_common_word():
    assert processor.most_common_word("the cat sat on the mat the") == "the"
    # tie -> first occurrence wins
    assert processor.most_common_word("a b b a") == "a"
    assert processor.most_common_word("") is None
```

A few notes on behavior:

- **`reverse_words`** uses `str.split()`, which splits on any whitespace run, so it collapses tabs/newlines too, not just spaces.
- **`is_palindrome`** keeps only alphanumeric characters (letters and digits); everything else — spaces, punctuation, symbols — is stripped before comparing. An empty string counts as a palindrome.
- **`caesar_cipher`** normalizes the shift with `shift %= 26`, so negative and multi-cycle shifts (e.g. 28) work correctly. Case is preserved per-letter (uppercase shifts to uppercase).
- **`most_common_word`** matches case-insensitively but returns the original casing of the word's first occurrence, and breaks ties in favor of whichever word appears earliest in the string.

Run the tests with:

```bash
pytest test_string_processor.py -v
```