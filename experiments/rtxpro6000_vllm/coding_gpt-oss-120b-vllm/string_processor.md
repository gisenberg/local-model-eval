**`string_processor.py`**

```python
import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """
    A small utility class that groups together a few common string‑processing
    operations. All methods are implemented as ``@staticmethod`` because they
    do not need to keep any state.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in *s* (the characters inside each word stay
        the same).  Consecutive whitespace characters are collapsed to a single
        space and any leading/trailing whitespace is removed.

        Example
        -------
        >>> StringProcessor.reverse_words("  hello   world  ")
        'world hello'
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Return the number of vowel letters (a, e, i, o, u) that appear in *s*.
        The check is case‑insensitive.

        Example
        -------
        >>> StringProcessor.count_vowels("OpenAI")
        4
        """
        return sum(1 for ch in s.lower() if ch in "aeiou")

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Return ``True`` if *s* reads the same forward and backward when
        ignoring case, spaces and any non‑alphanumeric characters
        (punctuation, symbols, etc.).

        Example
        -------
        >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
        True
        """
        # keep only alphanumeric characters and lower‑case them
        cleaned = re.sub(r"[^a-z0-9]", "", s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* with the given *shift*.  Only the letters
        ``a‑z`` and ``A‑Z`` are shifted; all other characters (digits,
        punctuation, whitespace, …) are left untouched.  ``shift`` may be
        negative or larger than 26 – it is wrapped around the alphabet.

        Example
        -------
        >>> StringProcessor.caesar_cipher("Abc-XYZ!", 3)
        'Def-ABC!'
        """
        def shift_char(ch: str) -> str:
            if "a" <= ch <= "z":
                base = ord("a")
                return chr((ord(ch) - base + shift) % 26 + base)
            if "A" <= ch <= "Z":
                base = ord("A")
                return chr((ord(ch) - base + shift) % 26 + base)
            return ch

        return "".join(shift_char(ch) for ch in s)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the word that occurs most frequently in *s* (case‑insensitive).
        Words are defined as consecutive alphanumeric characters
        (``\\w+``).  If several words share the highest frequency, the one that
        appears first in the original string is returned.  ``None`` is returned
        for an empty or whitespace‑only input.

        Example
        -------
        >>> StringProcessor.most_common_word("Apple banana apple")
        'apple'
        """
        # Normalise the string and extract words
        words = re.findall(r"\b\w+\b", s.lower())
        if not words:
            return None

        counts = Counter(words)
        max_freq = max(counts.values())

        # Preserve original order to break ties
        for word in words:
            if counts[word] == max_freq:
                return word
        return None   # pragma: no cover (logic never reaches here)
```

---

**`test_string_processor.py`**

```python
import pytest
from string_processor import StringProcessor


def test_reverse_words():
    assert StringProcessor.reverse_words("  Hello   world  ") == "world Hello"
    assert StringProcessor.reverse_words("one") == "one"
    assert StringProcessor.reverse_words("") == ""


def test_count_vowels():
    assert StringProcessor.count_vowels("OpenAI ChatGPT") == 7  # O e A I a a
    assert StringProcessor.count_vowels("bcdfg") == 0
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False


def test_caesar_cipher():
    # positive shift
    assert StringProcessor.caesar_cipher("Abc-XYZ!", 3) == "Def-ABC!"
    # negative shift
    assert StringProcessor.caesar_cipher("Def-ABC!", -3) == "Abc-XYZ!"
    # shift larger than alphabet size
    assert StringProcessor.caesar_cipher("xyz", 30) == "bcd"


def test_most_common_word():
    # normal case
    assert StringProcessor.most_common_word(
        "Apple banana apple orange banana apple"
    ) == "apple"

    # tie – first occurrence wins
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"

    # empty / whitespace only
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("   ") is None
```

Run the tests with:

```bash
pytest test_string_processor.py
```

All five tests exercise the corresponding methods of `StringProcessor`.