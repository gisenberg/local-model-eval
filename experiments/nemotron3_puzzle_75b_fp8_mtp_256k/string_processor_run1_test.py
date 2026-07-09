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