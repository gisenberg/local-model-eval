import re



class StringProcessor:
    """A utility class for processing and manipulating strings."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in a string, case-insensitive."""
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Keep only alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies Caesar cipher shift to alphabetic characters.
        Preserves case and non-alphabetic characters. Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Calculate shift using modulo 26 to handle large or negative numbers
                shifted = (ord(char) - start + shift) % 26
                result.append(chr(start + shifted))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Returns the most frequent word in a string (case-insensitive).
        If there is a tie, returns the word that appeared first.
        """
        if not s.strip():
            return None
        
        # Remove punctuation and split into words
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        # max() in Python is stable; it returns the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])


# --- Pytest Tests ---
import pytest

def test_string_processor():
    sp = StringProcessor()

    # Test reverse_words
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  Python is  fun  ") == "fun is Python"

    # Test count_vowels
    assert sp.count_vowels("Apple") == 2
    assert sp.count_vowels("Sky") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

    # Test is_palindrome
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False

    # Test caesar_cipher
    assert sp.caesar_cipher("Abc", 1) == "Bcd"
    assert sp.caesar_cipher("xyz", 3) == "abc"
    assert sp.caesar_cipher("Hello!", -1) == "Gdkkn!"

    # Test most_common_word
    assert sp.most_common_word("The cat sat on the mat") == "the"
    assert sp.most_common_word("Apple banana apple banana") == "apple" # Tie: apple comes first
    assert sp.most_common_word("") is None
    assert sp.most_common_word("!!!") is None

if __name__ == "__main__":
    # This allows running the tests directly via python file.py
    pytest.main([__file__])