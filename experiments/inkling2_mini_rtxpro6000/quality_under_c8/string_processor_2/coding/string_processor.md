Here's the `StringProcessor` class and the pytest tests:

**StringProcessor.py**

```python
from typing import Optional


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string (not characters).
        Multiple/leading/trailing spaces are normalized to single spaces
        (leading/trailing spaces removed)."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s if c.lower() in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces,
        and punctuation (alphanumeric characters only)."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher. Only a-z and A-Z are shifted (preserving
        case); other characters are unchanged. Supports negative shifts."""
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
        """Return the most frequently occurring word, case-insensitive.
        Ties are broken by first appearance. Returns None for empty strings.
        The returned word is in lowercase."""
        words = s.split()
        if not words:
            return None
        counts = {}
        for word in words:
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
        # dict iteration preserves first-appearance order, and the strict
        # '>' comparison means the earliest-appearing word wins ties.
        best, best_count = None, -1
        for key, count in counts.items():
            if count > best_count:
                best, best_count = key, count
        return best
```

**test_string_processor.py**

```python
from StringProcessor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  foo  ") == "foo world hello"
    assert sp.reverse_words("single") == "single"


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("XYZ") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("not a palindrome") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abcXYZ", -1) == "zabWXY"


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("apple banana apple banana") == "apple"
    assert sp.most_common_word("") is None
```

Notes on design choices:

- **`reverse_words`** uses `split()`, which splits on any whitespace runs — this handles multiple spaces, leading/trailing spaces, and tabs/newlines in one step.
- **`is_palindrome`** keeps only alphanumeric characters (so digits are counted as valid letters; everything else — punctuation, spaces, symbols — is ignored). The empty string is considered a palindrome.
- **`caesar_cipher`** relies on Python's modulo, which always returns a non-negative result for a positive modulus, so negative shifts work directly without extra normalization.
- **`most_common_word`** counts case-insensitively but returns the lowercase form of the word. Tie-breaking by first appearance works because dicts preserve insertion order — the `strict '>'` comparison keeps the earliest-seen word among equal counts.

Run with `pytest` from the directory containing the files.