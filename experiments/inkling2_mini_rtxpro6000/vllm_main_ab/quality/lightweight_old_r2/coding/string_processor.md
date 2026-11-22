Here's the implementation, followed by the tests.

**StringProcessor.py**

```python
from typing import Optional


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s (not the characters).

        Multiple spaces collapse to a single space; leading/trailing
        whitespace is removed.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the vowels (a, e, i, o, u) in s, case-insensitively."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check whether s is a palindrome, ignoring case, spaces,
        and punctuation."""
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift a-z/A-Z by shift positions (wraps around). All other
        characters are left unchanged. Negative shifts are supported."""
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

        Words are whitespace-separated tokens. On ties, the word whose
        first occurrence comes earliest is returned (with its original
        casing). Returns None for empty strings.
        """
        words = s.split()
        if not words:
            return None
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for i, w in enumerate(words):
            key = w.lower()
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, i)
        max_count = max(counts.values())
        best = min((k for k, c in counts.items() if c == max_count),
                   key=lambda k: first_seen[k])
        return words[first_seen[best]]
```

**test_string_processor.py**

```python
from StringProcessor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("one two three") == "three two one"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("xyz") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("ABC", 26) == "ABC"  # full rotation is a no-op


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # tie -> first occurrence
    assert sp.most_common_word("Hello hello HELLO world") == "Hello"  # keeps original casing
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

Run with:

```bash
pytest test_string_processor.py -v
```

A few design notes:

- `reverse_words` relies on `str.split()`, which naturally handles runs of whitespace and leading/trailing spaces.
- `is_palindrome` keeps only alphanumeric characters, so punctuation and spaces are ignored while digits (e.g., `"A1"`) are still compared.
- `caesar_cipher` uses `% 26` on the shift, so negative shifts and shifts ≥ 26 work correctly without special-casing.
- `most_common_word` treats whitespace-separated tokens as words (so punctuation attached to a word stays attached) and returns the word with the casing of its first occurrence.

If you'd rather strip punctuation in `most_common_word` too, that's a one-line change — just let me know.