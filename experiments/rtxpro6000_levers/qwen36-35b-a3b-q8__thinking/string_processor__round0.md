Here's the complete implementation with type hints, docstrings, and pytest tests:

### `string_processor.py`
```python
import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string. Multiple spaces become single spaces, leading/trailing spaces removed."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for char in s if char.lower() in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift. Only shifts a-z and A-Z, leaves others unchanged. Supports negative shifts."""
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive). Ties resolved by first appearance. Returns None for empty strings."""
        words = s.lower().split()
        if not words:
            return None

        counts = {}
        first_idx = {}
        for i, word in enumerate(words):
            counts[word] = counts.get(word, 0) + 1
            if word not in first_idx:
                first_idx[word] = i

        # Higher count wins; for ties, smaller index (first appearance) wins
        return max(counts, key=lambda w: (counts[w], -first_idx[w]))
```

### `test_string_processor.py`
```python
import pytest
from string_processor import StringProcessor

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("the sky is blue") == "blue is sky the"
    assert sp.reverse_words("  hello   world  ") == "world hello"
    assert sp.reverse_words("a") == "a"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("hello") == 2
    assert sp.count_vowels("AEIOU") == 5
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("rhythm") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("ABC", 1) == "BCD"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("abc", -1) == "zab"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the sky is blue") == "the"
    assert sp.most_common_word("apple banana apple cherry") == "apple"
    assert sp.most_common_word("a b a b") == "a"  # Tie broken by first appearance
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   ") is None
```

### How to Run
1. Save the class in `string_processor.py`
2. Save the tests in `test_string_processor.py`
3. Install pytest if needed: `pip install pytest`
4. Run tests: `pytest test_string_processor.py -v`

All methods include proper type hints, concise docstrings, and handle edge cases as specified. The Caesar cipher correctly leverages Python's modulo operator to handle negative shifts, and `most_common_word` uses a custom key to guarantee first-appearance tie-breaking.