```python
import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words, collapsing multiple spaces and trimming."""
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = set('aeiou')
        return sum(1 for ch in s.lower() if ch in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to letters only."""
        result = []
        for ch in s:
            if 'a' <= ch <= 'z':
                new_ord = (ord(ch) - ord('a') + shift) % 26 + ord('a')
                result.append(chr(new_ord))
            elif 'A' <= ch <= 'Z':
                new_ord = (ord(ch) - ord('A') + shift) % 26 + ord('A')
                result.append(chr(new_ord))
            else:
                result.append(ch)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return most frequent word (case-insensitive), ties broken by first occurrence."""
        if not s.strip():
            return None
        # Extract alphanumeric words (handles punctuation)
        words = re.findall(r'[a-zA-Z0-9]+', s.lower())
        if not words:
            return None
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        max_count = max(counts.values())
        # First word with max count (preserves original order)
        for word in words:
            if counts[word] == max_count:
                return word
        return None  # unreachable
```

```python
import pytest
from StringProcessor import StringProcessor

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("Hello world") == "world Hello"
    assert sp.reverse_words("  multiple   spaces  here ") == "here spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOUaeiou") == 10
    assert sp.count_vowels("bcdfg") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") == True
    assert sp.is_palindrome("race a car") == False
    assert sp.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("XYZ", 2) == "ZAB"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple cherry banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # tie, first occurrence
    assert sp.most_common_word("") is None
```