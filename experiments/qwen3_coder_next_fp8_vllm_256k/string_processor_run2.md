

Here's a complete implementation of the `StringProcessor` class with type hints, docstrings, and 5 pytest tests.

```python
import re
from typing import Optional
from collections import Counter


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string, preserving internal word spacing.

        Args:
            s: Input string.

        Returns:
            String with words in reverse order.

        Example:
            >>> reverse_words("Hello world from Python")
            "Python from world Hello"
        """
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in a string, case-insensitive.

        Args:
            s: Input string.

        Returns:
            Count of vowels (both uppercase and lowercase).
        """
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string.

        Returns:
            True if the string is a palindrome, False otherwise.

        Example:
            >>> is_palindrome("A man, a plan, a canal: Panama")
            True
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to the input string, shifting letters only.
        Non-alphabetic characters remain unchanged.
        Supports negative shifts (wrapping around alphabet).

        Args:
            s: Input string.
            shift: Number of positions to shift (can be negative or > 26).

        Returns:
            Ciphered string with only alphabetic characters shifted.

        Example:
            >>> caesar_cipher("abc XYZ", 3)
            "def ABC"
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
        
        for char in s:
            if 'a' <= char <= 'z':
                # Lowercase letters
                offset = ord('a')
                result.append(chr((ord(char) - offset + shift) % 26 + offset))
            elif 'A' <= char <= 'Z':
                # Uppercase letters
                offset = ord('A')
                result.append(chr((ord(char) - offset + shift) % 26 + offset))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).
        Words are defined as sequences of alphanumeric characters.
        In case of a tie, return the first word that reached the max count.

        Args:
            s: Input string.

        Returns:
            Most common word in lowercase, or None if no words found.
        """
        # Extract words: sequences of alphanumeric characters
        words = re.findall(r'\w+', s, re.ASCII)
        if not words:
            return None

        # Normalize to lowercase and count
        words_lower = [w.lower() for w in words]
        counter = Counter(words_lower)

        # Get max frequency
        max_freq = max(counter.values())
        
        # Return first word (original order) that has max frequency
        for word in words_lower:
            if counter[word] == max_freq:
                return word
        return None  # Shouldn't happen, but for type safety


# Example usage / quick sanity checks
if __name__ == "__main__":
    sp = StringProcessor()
    print(sp.reverse_words("Hello world from Python"))  # "Python from world Hello"
    print(sp.count_vowels("Hello World"))  # 3
    print(sp.is_palindrome("A man, a plan, a canal: Panama"))  # True
    print(sp.caesar_cipher("abc XYZ", 3))  # "def ABC"
    print(sp.most_common_word("Hello hello world!"))  # "hello"
```

---

### ✅ Pytest Tests (`test_string_processor.py`)

Create this file alongside your implementation and run with `pytest`:

```python
import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


class TestStringProcessor:
    def test_reverse_words(self, processor):
        assert processor.reverse_words("Hello world") == "world Hello"
        assert processor.reverse_words("a b c") == "c b a"
        assert processor.reverse_words("") == ""
        assert processor.reverse_words("   ") == ""  # Only spaces → empty list
        assert processor.reverse_words("single") == "single"

    def test_count_vowels(self, processor):
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("HELLO") == 2
        assert processor.count_vowels("xyz") == 0
        assert processor.count_vowels("AEIOUaeiou") == 10
        assert processor.count_vowels("") == 0

    def test_is_palindrome(self, processor):
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert processor.is_palindrome("No 'x' in Nixon") is True
        assert processor.is_palindrome("Hello, World!") is False
        assert processor.is_palindrome("RaceCar") is True
        assert processor.is_palindrome("") is True  # Empty string is palindrome

    def test_caesar_cipher(self, processor):
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("xyz", 3) == "abc"
        assert processor.caesar_cipher("ABC", -1) == "ZAB"
        assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
        assert processor.caesar_cipher("test", 26) == "test"
        assert processor.caesar_cipher("test", 52) == "test"
        assert processor.caesar_cipher("test", -26) == "test"

    def test_most_common_word(self, processor):
        assert processor.most_common_word("Hello hello world") == "hello"
        assert processor.most_common_word("apple banana apple banana") == "apple"  # tie → first
        assert processor.most_common_word("") is None
        assert processor.most_common_word("!!! ???") is None
        assert processor.most_common_word("The quick brown fox") == "the"  # case-insensitive
```

---

### 🧪 Run Tests

Install pytest if needed:

```bash
pip install pytest
```

Run:

```bash
pytest test_string_processor.py
```

Expected output: **5 passed**.

Let me know if you'd like the tests extended (e.g., edge cases, performance) or a CLI demo!