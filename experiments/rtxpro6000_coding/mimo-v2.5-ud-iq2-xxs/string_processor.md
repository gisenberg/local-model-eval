```python
from collections import Counter
from typing import Optional
import re


class StringProcessor:
    """A utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.

        Multiple spaces between words are collapsed to a single space.
        Leading and trailing spaces are removed.

        Args:
            s: The input string.

        Returns:
            A string with words in reversed order.
        """
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in a string.

        Vowels are a, e, i, o, u (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The count of vowels in the string.
        """
        return sum(1 for c in s if c.lower() in "aeiou")

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome.

        Ignores case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with the given shift.

        Only shifts letters a-z and A-Z. Other characters remain unchanged.
        Supports negative shifts.

        Args:
            s: The input string.
            shift: The number of positions to shift.

        Returns:
            The encrypted string.
        """
        result = []
        for c in s:
            if "a" <= c <= "z":
                result.append(chr((ord(c) - ord("a") + shift) % 26 + ord("a")))
            elif "A" <= c <= "Z":
                result.append(chr((ord(c) - ord("A") + shift) % 26 + ord("A")))
            else:
                result.append(c)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word in the string.

        Case-insensitive. If tied, returns the word that appears first.
        Returns None for empty strings.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty.
        """
        if not s.strip():
            return None
        words = s.lower().split()
        if not words:
            return None
        counts = Counter(words)
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None
```

```python
import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


class TestReverseWords:
    def test_reverse_words_basic(self, processor):
        assert processor.reverse_words("hello world") == "world hello"

    def test_reverse_words_multiple_spaces(self, processor):
        assert processor.reverse_words("hello   world") == "world hello"

    def test_reverse_words_leading_trailing(self, processor):
        assert processor.reverse_words("  hello world  ") == "world hello"

    def test_reverse_words_single_word(self, processor):
        assert processor.reverse_words("hello") == "hello"

    def test_reverse_words_empty(self, processor):
        assert processor.reverse_words("") == ""


class TestCountVowels:
    def test_count_vowels_basic(self, processor):
        assert processor.count_vowels("hello") == 2

    def test_count_vowels_uppercase(self, processor):
        assert processor.count_vowels("HELLO") == 2

    def test_count_vowels_mixed(self, processor):
        assert processor.count_vowels("Hello World") == 3

    def test_count_vowels_no_vowels(self, processor):
        assert processor.count_vowels("rhythm") == 0

    def test_count_vowels_empty(self, processor):
        assert processor.count_vowels("") == 0


class TestIsPalindrome:
    def test_is_palindrome_true(self, processor):
        assert processor.is_palindrome("racecar") is True

    def test_is_palindrome_false(self, processor):
        assert processor.is_palindrome("hello") is False

    def test_is_palindrome_case_insensitive(self, processor):
        assert processor.is_palindrome("RaceCar") is True

    def test_is_palindrome_with_spaces(self, processor):
        assert processor.is_palindrome("A man a plan a canal Panama") is True

    def test_is_palindrome_with_punctuation(self, processor):
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True


class TestCaesarCipher:
    def test_caesar_cipher_basic(self, processor):
        assert processor.caesar_cipher("abc", 3) == "def"

    def test_caesar_cipher_wrap(self, processor):
        assert processor.caesar_cipher("xyz", 3) == "abc"

    def test_caesar_cipher_negative(self, processor):
        assert processor.caesar_cipher("def", -3) == "abc"

    def test_caesar_cipher_mixed(self, processor):
        assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btqi!"

    def test_caesar_cipher_non_alpha(self, processor):
        assert processor.caesar_cipher("123!@#", 5) == "123!@#"


class TestMostCommonWord:
    def test_most_common_word_basic(self, processor):
        assert processor.most_common_word("the cat sat on the mat") == "the"

    def test_most_common_word_tied(self, processor):
        assert processor.most_common_word("cat dog cat dog") == "cat"

    def test_most_common_word_case_insensitive(self, processor):
        assert processor.most_common_word("The the THE") == "The"

    def test_most_common_word_empty(self, processor):
        assert processor.most_common_word("") is None

    def test_most_common_word_whitespace(self, processor):
        assert processor.most_common_word("   ") is None
```

**Key design decisions:**

- **`reverse_words`**: Uses `split()` which handles multiple spaces and strips leading/trailing whitespace automatically, then joins with a single space.
- **`count_vowels`**: Simple generator expression checking membership in `"aeiou"` after lowercasing.
- **`is_palindrome`**: Uses regex to strip non-alphanumeric characters, then compares the cleaned string to its reverse.
- **`caesar_cipher`**: Uses modular arithmetic (`% 26`) to wrap around the alphabet, supporting negative shifts naturally.
- **`most_common_word`**: Uses `Counter` for frequency counting, then iterates through the original word list to break ties by first appearance.