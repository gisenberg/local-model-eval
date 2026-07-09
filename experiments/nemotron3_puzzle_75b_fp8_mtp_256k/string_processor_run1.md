We need to output Python class StringProcessor with methods described, type hints, docstrings, and then include 5 pytest tests.

We should produce code block with the class definition and test functions.

Make sure to import necessary modules: re, collections.

Implement reverse_words: split by whitespace? Usually reverse order of words, preserving characters within each word? Likely "reverse_words" returns string with words reversed order: "Hello world" -> "world Hello". We'll assume split on whitespace, then join reversed list.

count_vowels: case-insensitive, count a e i o u.

is_palindrome: ignore case, spaces, punctuation. So remove non-alphanumeric? Usually letters only. We'll filter only alphanumeric and lower.

caesar_cipher: shift letters only (a-z, A-Z), preserve case, wrap around, support negative shift. Use modulo 26.

most_common_word: case-insensitive, return the word (original case? Usually lower? We'll return lowercased version). If tied, return the first occurring (in original order). We'll split on whitespace, strip punctuation? Probably simple: words separated by whitespace. We'll treat punctuation as part of word? Usually we ignore punctuation: maybe strip punctuation. For simplicity, we can use regex to find words: \b\w+\b. Then count lowercased, find max count, then iterate original list to get first word with that count, return lowercased version.

We'll include type hints: s: str, shift: int -> str; returns.

Docstrings for each method.

Then include test functions: test_reverse_words, test_count_vowels, test_is_palindrome, test_caesar_cipher, test_most_common_word. Use assert.

Make sure to import StringProcessor.

Write code in a single block with class and tests.

Let's craft.





```python
import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation tasks."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Return a new string with the order of words reversed.

        Words are defined as substrings separated by whitespace.
        The characters inside each word are left unchanged.

        Example:
            >>> StringProcessor.reverse_words("Hello   world")
            'world Hello'
        """
        # split on any whitespace, collapse multiple spaces
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in the string, case‑insensitive.

        Example:
            >>> StringProcessor.count_vowels("AbEcIdO")
            5
        """
        return sum(1 for ch in s.lower() if ch in "aeiou")

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Determine whether the string is a palindrome, ignoring case,
        spaces, and punctuation.

        Only alphanumeric characters are considered.

        Example:
            >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
            True
        """
        cleaned = re.sub(r"[^A-Za-z0-9]", "", s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to the string, shifting only alphabetic characters.

        * Upper‑case letters stay upper‑case, lower‑case stay lower‑case.
        * Non‑alphabetic characters are left unchanged.
        * The shift may be positive or negative and wraps around the alphabet.

        Example:
            >>> StringProcessor.caesar_cipher("Hello, World!", 3)
            'Khoor, Zruog!'
            >>> StringProcessor.caesar_cipher("Khoor, Zruog!", -3)
            'Hello, World!'
        """
        result = []
        for ch in s:
            if "a" <= ch <= "z":
                base = ord("a")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            elif "A" <= ch <= "Z":
                base = ord("A")
                result.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Return the most frequently occurring word in the string, case‑insensitive.
        If several words share the highest frequency, the word that appears
        first in the original string is returned.

        Words are sequences of alphanumeric characters; punctuation is ignored.
        Returns ``None`` for an empty or whitespace‑only string.

        Example:
            >>> StringProcessor.most_common_word("Apple banana apple orange banana")
            'apple'
        """
        # Extract words (alphanumeric sequences)
        words = re.findall(r"\b\w+\b", s)
        if not words:
            return None

        lowered = [w.lower() for w in words]
        freq = Counter(lowered)
        max_count = max(freq.values())

        # Return the first word (in original order) whose count equals max_count
        for original, low in zip(words, lowered):
            if freq[low] == max_count:
                return low.lower()  # return in lower‑case as per case‑insensitive spec
        return None


# ------------------------------
# Pytest tests
# ------------------------------

def test_reverse_words():
    assert StringProcessor.reverse_words("Hello   world") == "world Hello"
    assert StringProcessor.reverse_words("") == ""
    assert StringProcessor.reverse_words("single") == "single"
    assert StringProcessor.reverse_words("a b c d") == "d c b a"


def test_count_vowels():
    assert StringProcessor.count_vowels("AbEcIdO") == 5
    assert StringProcessor.count_vowels("Python") == 1
    assert StringProcessor.count_vowels("BCDFG") == 0
    assert StringProcessor.count_vowels("") == 0


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("No lemon, no melon") is True
    assert StringProcessor.is_palindrome("Hello") is False
    assert StringProcessor.is_palindrome("") is True  # empty string is trivially a palindrome
    assert StringProcessor.is_palindrome("Was it a car or a cat I saw?") is True


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert StringProcessor.caesar_cipher("Khoor, Zruog!", -3) == "Hello, World!"
    assert StringProcessor.caesar_cipher("abcXYZ", 2) == "cdeZAB"
    assert StringProcessor.caesar_cipher("abcXYZ", -2) == "yzaVWX"
    assert StringProcessor.caesar_cipher("123! @#", 5) == "123! @#"  # non‑letters unchanged


def test_most_common_word():
    s = "Apple banana apple orange banana"
    assert StringProcessor.most_common_word(s) == "apple"
    assert StringProcessor.most_common_word("one two three two one") == "one"
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("!!! ???") is None
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # tie -> first appearing
```

This file provides the `StringProcessor` class with the requested methods, complete with type hints, docstrings, and five corresponding pytest test cases. Feel free to save it (e.g., as `string_processor.py`) and run `pytest` to verify the implementation.