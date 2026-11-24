Here's the implementation and tests. I'll split them into two files: `string_processor.py` and `test_string_processor.py`.

**string_processor.py**

```python
"""String manipulation utilities."""

from typing import Optional


class StringProcessor:
    """Stateless utility methods for string processing."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words. Multiple spaces collapse to one;
        leading/trailing spaces are removed."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for ch in s.lower() if ch in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces,
        and punctuation."""
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift each letter by `shift` positions (wraps around the alphabet).
        Only a-z and A-Z are shifted; other characters are unchanged.
        Supports negative shifts."""
        result = []
        for ch in s:
            if "a" <= ch <= "z":
                result.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
            elif "A" <= ch <= "Z":
                result.append(chr((ord(ch) - ord("A") + shift) % 26 + ord("A")))
            else:
                result.append(ch)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        Ties are broken by first occurrence. Returns None for empty input."""
        words = s.split()
        if not words:
            return None
        counts = {}
        for w in words:
            key = w.lower()
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values())
        for w in words:  # first occurrence among words with max_count
            if counts[w.lower()] == max_count:
                return w
        return None
```

**test_string_processor.py**

```python
"""Tests for StringProcessor."""

from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("one two three") == "three two one"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello, World!") == 3  # e, o, o
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("rhythm") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abcXYZ 123!", 3) == "defABC 123!"
    assert sp.caesar_cipher("abc", -3) == "xyz"  # negative shift
    assert sp.caesar_cipher("abc", 26) == "abc"  # full cycle is identity


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("The cat sat on the mat") == "The"  # 'the' x2, returns first occurrence
    assert sp.most_common_word("apple banana apple banana") == "apple"  # tie -> first occurrence
    assert sp.most_common_word("") is None
```

Run with `pytest test_string_processor.py`.

A few design notes:

- `reverse_words` uses `str.split()` (no arguments), which splits on any whitespace runs — that handles multiple spaces, leading/trailing spaces, and empty input all at once.
- `is_palindrome` keeps only alphanumeric characters, so punctuation and spaces are dropped while digits are retained (e.g. `"A1B2"` is not a palindrome, but `"A1"` isn't either — fine either way, just noting digits are kept).
- `caesar_cipher` relies on Python's `%` operator, which returns a non-negative result even for negative operands, so negative shifts work naturally.
- `most_common_word` counts case-insensitively but returns the original-cased word from its first occurrence. Ties are resolved by scanning the original word list and returning the first word whose frequency equals the max — this guarantees "first appearance wins" even if a later word has the same count.

One limitation to be aware of: `most_common_word` treats punctuation as part of a word (`"cat,"` ≠ `"cat"`). If you want punctuation stripped too, change `s.split()` to a regex like `re.findall(r"\w+", s.lower())` and adjust the tie-breaking scan accordingly.