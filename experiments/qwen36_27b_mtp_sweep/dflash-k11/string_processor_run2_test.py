import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the string.
        
        Multiple consecutive whitespace characters are normalized to a single space.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Vowels are defined as a, e, i, o, u.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Only alphanumeric characters are considered for the comparison.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters.
        
        Preserves case and leaves non-alphabetic characters unchanged.
        Supports positive and negative shift values.
        """
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        
        Returns the first encountered word in case of a tie.
        Returns None if the string contains no words.
        """
        # Extract alphabetic words only
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        max_count = 0
        result: Optional[str] = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            # Strict greater than ensures we keep the first word in case of a tie
            if counts[word] > max_count:
                max_count = counts[word]
                result = word
        return result


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU") == 5
    assert StringProcessor.count_vowels("") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True
    assert StringProcessor.is_palindrome("") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("123 !@#", 5) == "123 !@#"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"  # Tie: cat appears first
    assert StringProcessor.most_common_word("123 !@#") is None
    assert StringProcessor.most_common_word("The THE the") == "the"