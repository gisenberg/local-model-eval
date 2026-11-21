Here's a complete implementation, split into the module and its tests.

**`string_processor.py`**

```python
import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s. Multiple internal spaces are
        collapsed to one, and leading/trailing spaces are removed."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Return True if s reads the same forwards and backwards,
        ignoring case, spaces, and punctuation (alphanumerics only)."""
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to s. Only a-z and A-Z are shifted;
        all other characters are left unchanged. Supports negative shifts."""
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
        """Return the most frequently occurring word in s, case-insensitively.
        Ties are broken by first appearance. Returns None for empty input.
        The returned word uses the original casing of its first occurrence."""
        words = s.split()
        if not words:
            return None
        counts = Counter(w.lower() for w in words)
        best = None
        best_count = -1
        for w in words:
            lw = w.lower()
            if counts[lw] > best_count:
                best_count = counts[lw]
                best = w
        return best
```

**`test_string_processor.py`**

```python
import pytest

from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words('  hello   world  python ') == 'python world hello'
    assert processor.reverse_words('') == ''
    assert processor.reverse_words('single') == 'single'


def test_count_vowels(processor):
    assert processor.count_vowels('Hello World') == 3
    assert processor.count_vowels('AEIOU') == 5
    assert processor.count_vowels('rhythm') == 0


def test_is_palindrome(processor):
    assert processor.is_palindrome('A man, a plan, a canal: Panama') is True
    assert processor.is_palindrome('race a car') is False
    assert processor.is_palindrome('') is True


def test_caesar_cipher(processor):
    assert processor.caesar_cipher('Hello, World!', 3) == 'Khoor, Zruog!'
    assert processor.caesar_cipher('abc', -1) == 'zab'
    assert processor.caesar_cipher('Z', 1) == 'A'  # wraps around


def test_most_common_word(processor):
    assert processor.most_common_word('the quick brown fox jumps over the lazy dog the') == 'the'
    assert processor.most_common_word('a b a b') == 'a'  # tie: first occurrence wins
    assert processor.most_common_word('') is None
```

Run with:

```bash
pytest test_string_processor.py -v
```

A few design notes:

- **`reverse_words`** uses `str.split()`, which already collapses any run of whitespace and strips the ends — no regex needed.
- **`is_palindrome`** keeps alphanumerics (digits are neither case nor punctuation), then compares the cleaned string to its reverse. An empty string counts as a palindrome, which is the conventional definition.
- **`caesar_cipher`** uses `% 26`, so negative shifts and shifts ≥ 26 work correctly with no special-casing. Case is preserved (uppercase letters wrap within A–Z).
- **`most_common_word`** counts case-insensitively but returns the original casing of the word. The tie-break works by scanning words in order and only replacing the winner when a *strictly* greater count is found — so the first-appearing word with the highest frequency wins.

One ambiguity worth flagging: for `most_common_word`, I chose to return the word with its original casing (from its first occurrence) rather than the lowercased form. If you'd rather always return lowercase, just change the `best = w` line to `best = lw`.