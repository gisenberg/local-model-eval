Here's the implementation and a matching test suite.

## `string_processor.py`

```python
from typing import Optional


class StringProcessor:
    """Utility methods for string manipulation and analysis."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in s (not the characters).

        Leading/trailing whitespace is removed and runs of multiple
        spaces between words are collapsed to a single space.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""
        return sum(1 for ch in s.lower() if ch in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Return True if s is a palindrome, ignoring case, spaces,
        and punctuation (alphanumeric characters only, compared in lowercase)."""
        cleaned = ''.join(ch for ch in s.lower() if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Return s with each a-z/A-Z letter shifted by `shift` positions
        in the alphabet (wrapping around). Non-letter characters are unchanged.
        Negative shifts move letters backward."""
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
        """Return the most frequently occurring word in s (case-insensitive).

        If there is a tie, the word that appears first in the string is
        returned (in its original form). Returns None for empty/whitespace-only
        strings.
        """
        words = s.split()
        if not words:
            return None

        counts = {}
        first_word = {}
        first_index = {}
        for i, word in enumerate(words):
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
            if key not in first_word:
                first_word[key] = word
                first_index[key] = i

        # Highest count wins; ties broken by earliest appearance.
        best_key = min(counts, key=lambda k: (-counts[k], first_index[k]))
        return first_word[best_key]
```

## `test_string_processor.py`

```python
from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  the   quick    brown fox  ") == "fox brown quick the"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("aeiouAEIOU") == 10
    assert sp.count_vowels("") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("not a palindrome") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 3) == "def"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("Hello, World!", -1) == "Gdkkn, Vnqlc!"


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple") == "apple"
    assert sp.most_common_word("The cat sat on the mat") == "The"
    assert sp.most_common_word("a b a b") == "a"  # tie -> first occurrence
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

## Notes

- **`is_palindrome`**: keeps alphanumeric characters only (punctuation and spaces dropped, case normalized), so "A man, a plan, a canal: Panama" is a palindrome.
- **`caesar_cipher`**: `shift %= 26` handles negative shifts and large shifts cleanly — e.g. `-1` shifts backward and `27` is equivalent to `1`.
- **`most_common_word`**: counting and tie-breaking are done case-insensitively, but the returned word preserves the casing of its first occurrence.
- Run the tests with:

```bash
pytest test_string_processor.py -v
```