Here's the implementation and the tests.

**`string_processor.py`**

```python
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s (not characters).

        Leading/trailing spaces are removed and runs of spaces between
        words are collapsed to a single space.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for ch in s.lower() if ch in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Return True if s reads the same forwards and backwards,
        ignoring case, spaces, and punctuation."""
        cleaned = ''.join(ch for ch in s.lower() if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift every a-z / A-Z letter by shift positions (wrapping around).

        Negative shifts shift backward. All other characters are unchanged.
        """
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

        If words tie, the one whose first occurrence appears earliest in s
        is returned. The returned word is lowercased. Returns None if s
        contains no words.
        """
        counts = {}
        order = []  # words in order of first appearance
        for word in s.split():
            w = word.lower()
            if w not in counts:
                counts[w] = 0
                order.append(w)
            counts[w] += 1

        best = None
        best_count = 0
        for w in order:
            if counts[w] > best_count:  # strict > keeps the earliest-appearing word
                best_count = counts[w]
                best = w
        return best
```

**`test_string_processor.py`**

```python
from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  the   quick\tbrown fox  ") == "fox brown quick the"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("rhythm") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True
    assert sp.is_palindrome("not a palindrome") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abcXYZ", -1) == "zabWXY"
    assert sp.caesar_cipher("abc", 26) == "abc"  # shift wraps around


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple cherry banana") == "apple"  # tie -> first occurrence
    assert sp.most_common_word("the the THE and and and") == "the"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

Notes on design choices:

- **`reverse_words`** uses `str.split()` (no arguments), which splits on any whitespace run and discards leading/trailing space automatically — exactly the required behavior.
- **`is_palindrome`** keeps alphanumerics (letters and digits) and drops everything else, so "Was it a car or a cat I saw?" passes.
- **`caesar_cipher`** takes the shift modulo 26, so negative shifts and shifts ≥ 26 work correctly.
- **`most_common_word`** splits on whitespace and lowercases for counting, then returns the lowercased word. The tie-breaking relies on iterating in first-appearance order with a strict `>` comparison, so the word that appears first in the string wins.

Run with `pytest test_string_processor.py -v`.