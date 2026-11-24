Here's the class and its tests. I've split them into two files, which is the standard layout for pytest.

**`string_processor.py`**

```python
import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """Utilities for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s.

        Multiple spaces between words collapse to a single space, and
        leading/trailing whitespace is removed.
        """
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for ch in s if ch.lower() in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Return True if s is a palindrome, ignoring case, spaces, punctuation."""
        cleaned = "".join(ch for ch in s if ch.isalnum()).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Return s with each letter shifted by shift positions (Caesar cipher).

        Only a-z / A-Z are shifted; other characters are left unchanged.
        Negative shifts are supported.
        """
        out = []
        for ch in s:
            if "a" <= ch <= "z":
                out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
            elif "A" <= ch <= "Z":
                out.append(chr((ord(ch) - ord("A") + shift) % 26 + ord("A")))
            else:
                out.append(ch)
        return "".join(out)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).

        Words are alphanumeric tokens (punctuation is ignored). Ties are
        broken by first appearance; the returned word uses the casing of
        its first occurrence. Returns None if s contains no words.
        """
        words = re.findall(r"[a-zA-Z0-9]+", s)
        if not words:
            return None
        counts = Counter(w.lower() for w in words)
        first_seen = {}
        for w in words:
            first_seen.setdefault(w.lower(), w)
        best_key, best_count = None, 0
        for w in words:  # iterate in order of appearance
            key = w.lower()
            if counts[key] > best_count:
                best_key, best_count = key, counts[key]
        return first_seen[best_key]
```

**`test_string_processor.py`**

```python
import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("  Hello   world  foo ") == "foo world Hello"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""


def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("rhythm") == 0


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False
    assert processor.is_palindrome("") is True


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"   # negative shift
    assert processor.caesar_cipher("xyz", 3) == "abc"    # wraps around
    assert processor.caesar_cipher("abc", 26) == "abc"   # full rotation


def test_most_common_word(processor):
    assert processor.most_common_word("the cat sat on the mat the") == "the"
    assert processor.most_common_word("dog cat dog cat") == "dog"  # tie -> first
    assert processor.most_common_word("Hello hello") == "Hello"   # casing of first
    assert processor.most_common_word("") is None
    assert processor.most_common_word("!!!") is None
```

Run with:

```bash
pytest test_string_processor.py -v
```

A few implementation notes:

- `reverse_words` uses `str.split()` (no arguments), which treats any run of whitespace as a single separator and strips leading/trailing whitespace — exactly the specified behavior.
- `is_palindrome` keeps only alphanumeric characters (`isalnum()`), so case, spaces, and punctuation are all ignored while digits are still compared.
- `caesar_cipher` uses modulo arithmetic on the offset, so any shift (positive, negative, or larger than 26) works uniformly.
- `most_common_word` tokenizes on alphanumeric runs so punctuation doesn't fragment words; it tracks `best_count` with a strict `>` while iterating in order of appearance, which makes first-appearance the tie-breaker. Returned words preserve the casing of their first occurrence — let me know if you'd rather it return a lowercased word.