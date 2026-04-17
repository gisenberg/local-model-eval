**`string_processor.py`**

```python
import re
from typing import Optional


class StringProcessor:
    """
    A collection of small string‑processing utilities.
    All methods are static because they do not need to keep any state.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in *s* (the characters inside each word stay the same).

        - Any amount of whitespace between words is collapsed to a single space.
        - Leading and trailing whitespace is removed.

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
        Return the number of vowel letters (a, e, i, o, u) in *s*,
        case‑insensitively.

        Example
        -------
        >>> StringProcessor.count_vowels("Hello")
        2
        """
        return sum(1 for ch in s.lower() if ch in "aeiou")

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Check whether *s* reads the same forward and backward,
        ignoring case, spaces and any non‑alphanumeric characters.

        Example
        -------
        >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
        True
        """
        # keep only letters and digits, lower‑cased
        cleaned = re.sub(r"[^a-z0-9]", "", s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to *s* with the given *shift*.
        Only alphabetic characters (A‑Z, a‑z) are shifted; everything else
        (digits, punctuation, whitespace…) stays unchanged.
        Negative shifts are supported.

        Example
        -------
        >>> StringProcessor.caesar_cipher("Abc XYZ!", 3)
        'Def ABC!'
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
        If several words share the highest frequency, the one that appears
        first in the original string is returned.
        Returns ``None`` for an empty or whitespace‑only string.

        Example
        -------
        >>> StringProcessor.most_common_word("apple banana apple")
        'apple'
        """
        # Normalise the string and extract words (alphanumerics + underscore)
        words = re.findall(r"\b\w+\b", s.lower())
        if not words:
            return None

        # Count while preserving first‑appearance order
        counts = {}
        order = []          # order of first appearance
        for w in words:
            if w not in counts:
                counts[w] = 0
                order.append(w)
            counts[w] += 1

        # Determine the maximum frequency
        max_freq = max(counts.values())
        # Return the first word (according to original order) that has max_freq
        for w in order:
            if counts[w] == max_freq:
                return w
        # The loop always returns; the line below is just for type‑checkers
        return None
```

---

**`test_string_processor.py`**

```python
import pytest
from string_processor import StringProcessor


def test_reverse_words():
    assert StringProcessor.reverse_words("  hello   world  ") == "world hello"
    assert StringProcessor.reverse_words("") == ""
    assert StringProcessor.reverse_words("single") == "single"
    assert StringProcessor.reverse_words("  a   b   c  ") == "c b a"


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("bcdfg") == 0
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False
    assert StringProcessor.is_palindrome("") is True   # empty string is a palindrome


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Abc XYZ!", 3) == "Def ABC!"
    assert StringProcessor.caesar_cipher("Abc XYZ!", -3) == "Xyz UVW!"
    assert StringProcessor.caesar_cipher("Hello, World!", 0) == "Hello, World!"
    # wrap‑around test
    assert StringProcessor.caesar_cipher("zZ", 1) == "aA"


def test_most_common_word():
    assert (
        StringProcessor.most_common_word(
            "apple banana apple orange banana apple"
        )
        == "apple"
    )
    # tie – the word that appears first should win
    assert StringProcessor.most_common_word("Cat dog cat Dog") == "cat"
    # empty / whitespace only strings
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("   ") is None
    # punctuation should be ignored because we split on word boundaries
    assert StringProcessor.most_common_word("Hello, hello! HELLO?") == "hello"
```

Run the tests with:

```bash
pytest test_string_processor.py
```

All tests should pass, confirming that each method behaves as specified.