import re



class StringProcessor:
    """A utility class for advanced string manipulations."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string while preserving whitespace."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels (a, e, i, o, u) in a string, case-insensitive."""
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        # Remove all non-alphanumeric characters and convert to lowercase
        clean_s = "".join(char.lower() for char in s if char.isalnum())
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Shifts letters by the given amount. 
        Only affects a-z and A-Z. Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Calculate new position using modulo 26
                shifted = (ord(char) - start + shift) % 26
                result.append(chr(start + shifted))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Returns the most frequent word in a string (case-insensitive).
        Returns the first occurring word in case of a tie.
        """
        if not s.strip():
            return None
        
        # Extract words using regex to handle punctuation
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        # max() is stable in Python, so it returns the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])


# --- Pytest Tests ---
import pytest

def test_string_processor():
    sp = StringProcessor()

    # Test reverse_words
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  Python is fun  ") == "fun is Python"

    # Test count_vowels
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("Sky") == 0

    # Test is_palindrome
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race car") is True
    assert sp.is_palindrome("hello") is False

    # Test caesar_cipher
    assert sp.caesar_cipher("Hello World!", 3) == "Khoor Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("xyz", 1) == "yza"

    # Test most_common_word
    assert sp.most_common_word("Apple banana apple orange") == "apple"
    assert sp.most_common_word("The cat and the dog") == "the"
    assert sp.most_common_word("   ") is None
    # Tie test: "apple" appears first, "banana" appears second. Both count 1.
    assert sp.most_common_word("apple banana") == "apple"

if __name__ == "__main__":
    # This allows running the tests directly via `python filename.py` 
    # though `pytest filename.py` is the standard way.
    pytest.main([__file__])