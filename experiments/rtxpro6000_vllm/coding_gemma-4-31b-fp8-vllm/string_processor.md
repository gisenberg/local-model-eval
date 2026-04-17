Here is the complete implementation of the `StringProcessor` class and the corresponding `pytest` test suite.

### Implementation: `string_processor.py`

```python
import re
from collections import Counter
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, normalizing spaces."""
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        clean_s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case."""
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position using modulo 26 to handle wrap-around and negative shifts
                new_char = chr(start + (ord(char) - start + shift) % 26)
                result.append(new_char)
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word (case-insensitive). Returns None if empty."""
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
        
        counts = Counter(words)
        # max() in Python is stable, so it returns the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])
```

---

### Tests: `test_string_processor.py`

To run these tests, ensure you have `pytest` installed (`pip install pytest`) and run the command `pytest test_string_processor.py` in your terminal.

```python
import pytest
from string_processor import StringProcessor

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    # Test basic reversal and space normalization
    assert processor.reverse_words("  Hello   World  ") == "World Hello"
    assert processor.reverse_words("Python is awesome") == "awesome is Python"

def test_count_vowels(processor):
    # Test case insensitivity and mixed characters
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU aeiou") == 10
    assert processor.count_vowels("xyz") == 0

def test_is_palindrome(processor):
    # Test ignoring case, spaces, and punctuation
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    # Test positive shift, negative shift, and non-alpha characters
    assert processor.caesar_cipher("Abc", 1) == "Bcd"
    assert processor.caesar_cipher("Bcd", -1) == "Abc"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert processor.caesar_cipher("zZ", 1) == "aA"

def test_most_common_word(processor):
    # Test frequency, case insensitivity, ties, and empty strings
    assert processor.most_common_word("Apple apple Banana") == "apple"
    assert processor.most_common_word("The cat and the dog") == "the"
    # Tie test: "apple" and "banana" both appear twice; "apple" comes first
    assert processor.most_common_word("apple banana apple banana") == "apple"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("!!!") is None
```

### Key Design Decisions:
1.  **`reverse_words`**: Used `.split()` without arguments, which automatically handles any amount of whitespace and strips leading/trailing spaces.
2.  **`is_palindrome`**: Used a Regular Expression (`[^a-zA-Z0-9]`) to strip everything except alphanumeric characters to ensure punctuation doesn't interfere.
3.  **`caesar_cipher`**: Used the modulo operator `% 26`. This is the mathematically robust way to handle both large positive shifts and negative shifts (Python's `%` operator handles negative numbers correctly for this use case).
4.  **`most_common_word`**: Used `re.findall(r'\w+', ...)` to extract words while ignoring punctuation. I used `max()` on the original `words` list rather than the `Counter` keys to ensure that in the event of a tie, the word that appeared **first** in the string is returned.