Here's a complete implementation of the `StringProcessor` class with type hints and docstrings, followed by 5 pytest tests covering each method.

---

### ✅ `StringProcessor` class

```python
from typing import Optional
import string
import re
from collections import Counter


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in the input string.
        Multiple spaces between words are collapsed to a single space.
        Leading and trailing spaces are removed.
        """
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in the string, case-insensitive.
        """
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        Only alphanumeric characters are considered.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to the input string with the given shift.
        Only alphabetic characters (a-z, A-Z) are shifted; others remain unchanged.
        Supports negative shifts (wraps around alphabet).
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
        for char in s:
            if 'a' <= char <= 'z':
                base = ord('a')
                shifted = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted)
            elif 'A' <= char <= 'Z':
                base = ord('A')
                shifted = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        Words are sequences of alphanumeric characters separated by non-alphanumeric characters.
        If multiple words tie in frequency, return the one that appears first.
        Returns None for empty strings or strings with no words.
        """
        if not s.strip():
            return None

        # Extract words: sequences of alphanumeric chars
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        # Count frequencies while preserving first occurrence order
        freq = Counter(words)
        max_count = max(freq.values())
        for word in words:
            if freq[word] == max_count:
                return word
        return None
```

---

### ✅ Pytest Tests (`test_string_processor.py`)

```python
import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  a  b   c  ") == "c b a"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("single") == "single"


def test_count_vowels(processor):
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0


def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("Hello") is False
    assert processor.is_palindrome("") is True  # empty string is palindrome


def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 3) == "abc"
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("Test123!", 26) == "Test123!"


def test_most_common_word(processor):
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("Apple apple BANANA") == "apple"
    assert processor.most_common_word("one two three") is None  # all tie → return first: "one"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None
    assert processor.most_common_word("word! word, word.") == "word"
```

---

### 📝 Notes

- `most_common_word` uses `\b\w+\b` to extract words (alphanumeric + underscore), which is standard for word tokenization in many contexts. If you want stricter word definition (e.g., only letters), replace `\w` with `[a-zA-Z]+`.
- Caesar cipher handles negative shifts by normalizing with modulo 26.
- All methods are robust to edge cases (empty strings, whitespace, punctuation, etc.).

Let me know if you'd like to run these tests or need a `pytest.ini` or `conftest.py` setup!