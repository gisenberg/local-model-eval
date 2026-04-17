Here's the complete implementation with type hints, docstrings, and pytest tests:

```python
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string. Multiple spaces become single spaces, leading/trailing spaces removed."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        return sum(1 for c in s if c.lower() in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift. Only shifts a-z and A-Z, leaves others unchanged. Supports negative shifts."""
        result = []
        for c in s:
            if c.isalpha():
                base = ord('A') if c.isupper() else ord('a')
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive). Ties resolved by first occurrence. Returns None for empty strings."""
        words = s.lower().split()
        if not words:
            return None
            
        counts = {}
        first_idx = {}
        for i, w in enumerate(words):
            counts[w] = counts.get(w, 0) + 1
            if w not in first_idx:
                first_idx[w] = i
                
        # Maximize count, then minimize index (achieved by negating index for tie-breaking)
        return max(counts.keys(), key=lambda w: (counts[w], -first_idx[w]))
```

### Pytest Tests

```python
import pytest

class TestStringProcessor:
    def setup_method(self):
        self.sp = StringProcessor()

    def test_reverse_words(self):
        assert self.sp.reverse_words("hello world") == "world hello"
        assert self.sp.reverse_words("  multiple   spaces  here  ") == "here spaces multiple"
        assert self.sp.reverse_words("single") == "single"
        assert self.sp.reverse_words("") == ""

    def test_count_vowels(self):
        assert self.sp.count_vowels("hello") == 2
        assert self.sp.count_vowels("AEIOU") == 5
        assert self.sp.count_vowels("bcdfg") == 0
        assert self.sp.count_vowels("rhythm") == 0

    def test_is_palindrome(self):
        assert self.sp.is_palindrome("A man, a plan, a canal: Panama") == True
        assert self.sp.is_palindrome("racecar") == True
        assert self.sp.is_palindrome("hello") == False
        assert self.sp.is_palindrome("Was it a car or a cat I saw?") == True

    def test_caesar_cipher(self):
        assert self.sp.caesar_cipher("abc", 1) == "bcd"
        assert self.sp.caesar_cipher("xyz", 3) == "abc"
        assert self.sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
        assert self.sp.caesar_cipher("abc", -1) == "zab"

    def test_most_common_word(self):
        assert self.sp.most_common_word("the cat sat on the mat") == "the"
        assert self.sp.most_common_word("apple banana apple") == "apple"
        assert self.sp.most_common_word("red blue green") == "red"  # Tie: first occurrence wins
        assert self.sp.most_common_word("") is None
        assert self.sp.most_common_word("   ") is None
```

### Key Implementation Details:
1. **`reverse_words`**: Uses `str.split()` without arguments, which automatically handles multiple spaces, strips leading/trailing whitespace, and returns a list of words. Reversing and joining with `" "` satisfies all spacing requirements.
2. **`count_vowels`**: Generator expression with `sum()` for memory efficiency. Case-insensitivity handled via `.lower()`.
3. **`is_palindrome`**: Filters only alphanumeric characters and lowercases them before comparing to its reverse.
4. **`caesar_cipher`**: Uses modulo arithmetic `(ord(c) - base + shift) % 26`. Python's `%` operator correctly handles negative shifts (e.g., `-1 % 26 == 25`), so no extra conditional logic is needed.
5. **`most_common_word`**: Tracks both frequency and first occurrence index. The `max()` key tuple `(count, -index)` elegantly resolves ties by preferring the smallest index (largest negative value).
6. **Tests**: Each method is covered with standard cases, edge cases (empty strings, ties, negative shifts, punctuation), and assertions match the specified behavior. Run with `pytest test_file.py`.