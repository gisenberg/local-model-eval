Here's the complete solution. Save it as two files: `string_processor.py` and `test_string_processor.py`.

**`string_processor.py`**

```python
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, collapsing multiple spaces to one."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        return sum(1 for c in s.lower() if c in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Check palindrome ignoring case, spaces, and punctuation."""
        cleaned = "".join(c for c in s.lower() if c.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift a-z/A-Z by `shift` (negative supported); leave other chars unchanged."""
        result = []
        for c in s:
            if 'a' <= c <= 'z':
                result.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= c <= 'Z':
                result.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(c)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Ties broken by first
        appearance. Returns None for empty/blank strings."""
        counts = {}
        for word in s.lower().split():
            counts[word] = counts.get(word, 0) + 1
        if not counts:
            return None
        max_count = max(counts.values())
        for word, count in counts.items():  # dicts preserve insertion order
            if count == max_count:
                return word
        return None  # unreachable, but satisfies the type checker
```

**`test_string_processor.py`**

```python
from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("one") == "one"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3  # e, o, o
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU") == 5


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 2) == "cde"
    assert sp.caesar_cipher("cde", -2) == "abc"       # negative shift
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("xyz", 3) == "abc"          # wraps around


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("b a a b") == "a"         # tie -> first seen
    assert sp.most_common_word("   ") is None            # empty -> None
```

Notes:

- **`reverse_words`**: Python's `str.split()` (no args) splits on any whitespace run and strips leading/trailing whitespace, so `[::-1]` and `" ".join(...)` produce exactly the required output.
- **`is_palindrome`**: Filters with `isalnum()`, which keeps letters and digits while dropping spaces and punctuation, then compares the result to its reverse.
- **`caesar_cipher`**: `% 26` on the shifted letter index handles wrap-around and negative shifts (Python's modulo always returns a non-negative result).
- **`most_common_word`**: Counts on lowercased words but preserves insertion order, so when counts tie, the word that first appeared wins. Note it returns the lowercase normalized form (e.g., `"The"` → `"the"`).
- Run the tests with `pytest test_string_processor.py -v`.