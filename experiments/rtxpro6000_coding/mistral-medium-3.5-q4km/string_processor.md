Here's the implementation of the `StringProcessor` class with all the requested methods, along with pytest tests:

```python
from typing import Optional
import re
from collections import defaultdict

class StringProcessor:
    """A class for processing strings with various operations."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string, handling multiple spaces.

        Args:
            s: Input string to process

        Returns:
            String with words in reverse order, single spaces between words,
            no leading/trailing spaces
        """
        words = s.split()
        return ' '.join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string to count vowels in

        Returns:
            Number of vowels (a, e, i, o, u) in the string
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in s.lower() if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string to check

        Returns:
            True if the string is a palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters.

        Args:
            s: Input string to encrypt
            shift: Number of positions to shift (can be negative)

        Returns:
            String with alphabetic characters shifted, other characters unchanged
        """
        result = []
        shift = shift % 26  # Normalize shift to handle large values

        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                new_ord = ord(char) + shift
                if new_ord > ord('z'):
                    new_ord -= 26
                elif new_ord < ord('a'):
                    new_ord += 26
                result.append(chr(new_ord))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                new_ord = ord(char) + shift
                if new_ord > ord('Z'):
                    new_ord -= 26
                elif new_ord < ord('A'):
                    new_ord += 26
                result.append(chr(new_ord))
            else:
                # Leave other characters unchanged
                result.append(char)

        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most frequently occurring word in a string (case-insensitive).

        Args:
            s: Input string to analyze

        Returns:
            The most common word (original case of first occurrence), or None if empty
        """
        if not s.strip():
            return None

        words = s.split()
        word_counts = defaultdict(int)
        first_occurrence = {}

        for i, word in enumerate(words):
            lower_word = word.lower()
            word_counts[lower_word] += 1
            if lower_word not in first_occurrence:
                first_occurrence[lower_word] = i

        if not word_counts:
            return None

        max_count = max(word_counts.values())
        candidates = [word for word, count in word_counts.items() if count == max_count]

        # Among candidates, find the one with the smallest first occurrence index
        # and return the original word (not lowercased)
        min_index = min(first_occurrence[word] for word in candidates)
        original_word = words[min_index]

        return original_word

# Pytest tests
import pytest

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces   ") == "spaces multiple"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("  leading and trailing  ") == "trailing and leading"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("AEIOUaeiou") == 10
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("The quick brown fox") == 5

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True
    assert processor.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("abc", 26) == "abc"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("hello hello world") == "hello"
    assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("  ") is None
    assert processor.most_common_word("apple banana apple banana") == "apple"
```

### Explanation:

1. **reverse_words**:
   - Splits the string into words (handling multiple spaces automatically)
   - Reverses the list of words and joins them with single spaces

2. **count_vowels**:
   - Converts the string to lowercase and counts characters that are in the vowels set

3. **is_palindrome**:
   - Removes all non-alphanumeric characters using regex
   - Compares the cleaned string with its reverse (case-insensitive)

4. **caesar_cipher**:
   - Handles both positive and negative shifts (normalized with modulo 26)
   - Only shifts alphabetic characters, preserving case
   - Leaves other characters unchanged

5. **most_common_word**:
   - Returns None for empty strings or strings with only whitespace
   - Uses a dictionary to count word occurrences (case-insensitive)
   - Tracks first occurrence indices to handle ties (returns the first occurring word)
   - Returns the original word (not lowercased) from the first occurrence

The pytest tests cover:
- Normal cases
- Edge cases (empty strings, single words)
- Special cases (punctuation, mixed case, ties in word frequency)
- Negative shifts for the cipher
- Multiple spaces handling