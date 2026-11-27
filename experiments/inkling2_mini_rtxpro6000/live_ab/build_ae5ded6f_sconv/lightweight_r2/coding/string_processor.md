Here's the implementation and tests:

**string_processor.py**

```python
from typing import Optional, Dict


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, collapsing multiple spaces into one."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for ch in s.lower() if ch in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(ch for ch in s.lower() if ch.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Shift a-z/A-Z by `shift` positions (wraps around); leave other chars unchanged."""
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
        """Return the most frequent word (case-insensitive).
        On a tie, return the one that appears first. Return None for empty strings."""
        words = s.split()
        if not words:
            return None
        counts: Dict[str, int] = {}
        for w in words:
            key = w.lower()
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values())
        for w in words:  # first occurrence wins ties
            if counts[w.lower()] == max_count:
                return w
        return None  # unreachable, but satisfies type checker
```

**test_string_processor.py**

```python
from string_processor import StringProcessor


def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("  hello   world  foo  ") == "foo world hello"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("xyz") == 0


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("not a palindrome") is False


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("Hello", -3) == "Ebiil"
    assert sp.caesar_cipher("abc", 26) == "abc"  # full wrap-around


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the quick brown fox the lazy dog the") == "the"
    assert sp.most_common_word("apple banana apple cherry banana") == "apple"  # tie -> first
    assert sp.most_common_word("   ") is None
```

**Notes:**
- `caesar_cipher` uses `% 26` on the shift, so negative shifts and shifts ≥ 26 work naturally.
- `most_common_word` compares words case-insensitively but returns the word with its original casing, and breaks ties by first appearance in the string.
- `is_palindrome` strips everything that isn't a letter or digit before comparing.

Run the tests with:

```bash
pytest test_string_processor.py -v
```