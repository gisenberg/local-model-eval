Here's the implementation and tests. I've split them into two files so the tests can import the class directly.

**`string_processor.py`**

```python
"""StringProcessor: string manipulation utilities."""
import re
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Return the words in s in reverse order, joined by single spaces.

        Leading/trailing whitespace is removed and runs of internal
        whitespace are collapsed to single spaces.
        """
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s (case-insensitive)."""
        return sum(1 for ch in s if ch.lower() in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Return True if s is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [ch.lower() for ch in s if ch.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Return s with each letter shifted by `shift` positions (wrapping around).

        Only a-z and A-Z are shifted; other characters are unchanged.
        Negative shifts are supported.
        """
        def shift_char(ch: str, base: int) -> str:
            return chr((ord(ch) - base + shift) % 26 + base)

        out = []
        for ch in s:
            if "a" <= ch <= "z":
                out.append(shift_char(ch, ord("a")))
            elif "A" <= ch <= "Z":
                out.append(shift_char(ch, ord("A")))
            else:
                out.append(ch)
        return "".join(out)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word in s (case-insensitive).

        Words are runs of letters/digits. Ties are broken by first occurrence
        in s; the returned word keeps its original casing. Returns None if s
        contains no words.
        """
        words = re.findall(r"[A-Za-z0-9]+", s)
        if not words:
            return None

        counts = {}
        first_seen = {}
        for i, word in enumerate(words):
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, i)

        # Highest count wins; ties broken by earliest first occurrence.
        best = min(counts, key=lambda k: (-counts[k], first_seen[k]))
        for word in words:
            if word.lower() == best:
                return word
        return None
```

**`test_string_processor.py`**

```python
"""Tests for StringProcessor. Run with: pytest test_string_processor.py"""
from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  the   quick\t brown\nfox ") == "fox brown quick the"
    assert sp.reverse_words("one") == "one"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("hello") == 2
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("Aeiou") == 5


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("") is True


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 3) == "def"
    assert sp.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
    assert sp.caesar_cipher("abc", -3) == "xyz"          # negative shift
    assert sp.caesar_cipher("xyz", 3) == "abc"           # wraps around
    assert sp.caesar_cipher("abcXYZ", 26) == "abcXYZ"    # full rotation = identity


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("apple banana apple cherry banana") == "apple"  # tie → first
    assert sp.most_common_word("Hello, world! hello?") == "Hello"             # original casing
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   !!!  ") is None
```

A few design decisions worth noting:

- **`reverse_words`** relies on `str.split()` with no arguments, which splits on any whitespace run, strips leading/trailing whitespace, and collapses internal runs — exactly what the spec asks for.
- **`is_palindrome`** keeps only `isalnum()` characters (letters and digits), so punctuation and spaces are ignored but digits are still compared. `isalnum()` also accepts Unicode letters.
- **`caesar_cipher`** uses `% 26` so negative shifts and shifts ≥ 26 (and multiples of 26) work correctly.
- **`most_common_word`** extracts words via regex (so punctuation doesn't create distinct "words" like `"hello,"`), counts case-insensitively, breaks ties by first occurrence, and returns the word as originally written (e.g. `"Hello"`). If you'd rather get the lowercase canonical form back, just `return best` after the tie-break — one-line change.

Run the tests with `pytest test_string_processor.py -v`.