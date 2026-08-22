

```python
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.

        Multiple spaces are collapsed to one; leading/trailing spaces are removed.
        """
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s.lower() if c in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = "".join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher with the given shift.

        Only a-z and A-Z are shifted (wrapping around); all other characters
        are left unchanged. Negative shifts are supported.
        """
        result: list[str] = []
        for c in s:
            if c.islower():
                shifted = (ord(c) - ord("a") + shift) % 26
                result.append(chr(shifted + ord("a")))
            elif c.isupper():
                shifted = (ord(c) - ord("A") + shift) % 26
                result.append(chr(shifted + ord("A")))
            else:
                result.append(c)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).

        If multiple words share the highest frequency, the one that appears
        first in the string is returned. Returns None for empty/whitespace-only strings.
        """
        if not s or not s.strip():
            return None
        words = s.lower().split()
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None  # unreachable if s is non-empty, but satisfies type checker


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
import pytest


class TestStringProcessor:
    """One focused test per public method."""

    def test_reverse_words(self) -> None:
        sp = StringProcessor()
        assert sp.reverse_words("  hello   world  foo  ") == "foo world hello"
        assert sp.reverse_words("single") == "single"
        assert sp.reverse_words("   ") == ""
        assert sp.reverse_words("") == ""

    def test_count_vowels(self) -> None:
        sp = StringProcessor()
        assert sp.count_vowels("Hello World") == 3  # e, o, o
        assert sp.count_vowels("AEIOU aeiou") == 10
        assert sp.count_vowels("rhythm myths") == 0
        assert sp.count_vowels("") == 0

    def test_is_palindrome(self) -> None:
        sp = StringProcessor()
        assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
        assert sp.is_palindrome("Was it a car or a cat I saw?") is True
        assert sp.is_palindrome("race a car") is False
        assert sp.is_palindrome("12321") is True
        assert sp.is_palindrome("") is True  # empty string is trivially a palindrome

    def test_caesar_cipher(self) -> None:
        sp = StringProcessor()
        assert sp.caesar_cipher("abc", 3) == "def"
        assert sp.caesar_cipher("xyz", 3) == "abc"       # wraps around
        assert sp.caesar_cipher("abc", -1) == "zab"      # negative shift
        assert sp.caesar_cipher("Hello, World! 123", 1) == "Ifmmp, Xpsme! 123"
        assert sp.caesar_cipher("abc", 0) == "abc"       # zero shift is identity
        assert sp.caesar_cipher("abc", 26) == "abc"      # full rotation

    def test_most_common_word(self) -> None:
        sp = StringProcessor()
        assert sp.most_common_word("the cat the dog the") == "the"
        assert sp.most_common_word("Hello hello HELLO world") == "hello"
        # tie → first occurrence wins
        assert sp.most_common_word("apple banana apple banana") == "apple"
        assert sp.most_common_word("") is None
        assert sp.most_common_word("   ") is None
```

### How to run

```bash
pytest test_string_processor.py -v
```

### Key design decisions

| Method | Detail |
|---|---|
| `reverse_words` | `str.split()` (no arg) handles any run of whitespace, so a single `" ".join` restores exactly one space between words. |
| `count_vowels` | Single pass with a generator expression; no regex needed. |
| `is_palindrome` | `str.isalnum()` strips punctuation *and* spaces in one check; comparison is done on the lowercased, cleaned string. |
| `caesar_cipher` | Python's `%` operator returns a non-negative result for negative operands (`-1 % 26 == 25`), so negative shifts "just work" without a special branch. |
| `most_common_word` | A first-appearance scan after counting guarantees the correct tie-break without needing a stable-sort trick. Returns `None` for both `""` and whitespace-only input. |