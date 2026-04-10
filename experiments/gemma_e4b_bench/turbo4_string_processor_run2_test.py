
import re


class StringProcessor:
    """
    A utility class for performing various string manipulation tasks.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string and collapses multiple spaces into one.

        Args:
            s: The input string.

        Returns:
            A string with the word order reversed and single spaces between words.
        """
        words = s.split()
        return " ".join(words[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels (a, e, i, o, u) in a string, case-insensitively.

        Args:
            s: The input string.

        Returns:
            The total count of vowels.
        """
        vowels = "aeiouAEIOU"
        count = 0
        for char in s:
            if char in vowels:
                count += 1
        return count

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned_s == cleaned_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies the Caesar cipher shift to alphabetic characters in a string.
        Non-alphabetic characters are left unchanged. Supports negative shifts.

        Args:
            s: The input string.
            shift: The integer shift value.

        Returns:
            The encrypted or decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Handle lowercase letters
                start = ord('a')
                new_ord = (ord(char) - start + shift) % 26 + start
                result.append(chr(new_ord))
            elif 'A' <= char <= 'Z':
                # Handle uppercase letters
                start = ord('A')
                new_ord = (ord(char) - start + shift) % 26 + start
                result.append(chr(new_ord))
            else:
                # Keep non-alphabetic characters as they are
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most frequently occurring word in a string (case-insensitive).
        If there is a tie, the word that appears first in the text is returned.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty or contains no words.
        """
        # Use regex to find all sequences of word characters
        words = re.findall(r'\b\w+\b', s.lower())
        
        if not words:
            return None

        # Count frequencies
        word_counts = Counter(words)
        
        # Find the maximum frequency
        max_count = max(word_counts.values())
        
        # Iterate through the original list of words to respect the "first if tied" rule
        for word in words:
            if word_counts[word] == max_count:
                return word
        
        # Should theoretically not be reached if words is not empty
        return None

# Example Usage (optional, for demonstration)
if __name__ == '__main__':
    processor = StringProcessor()
    
    print("--- Word Reversal ---")
    print(f"Original: 'Hello   world, how are you?' -> {processor.reverse_words('Hello   world, how are you?')}")

    print("\n--- Vowel Count ---")
    print(f"String: 'Programming' -> {processor.count_vowels('Programming')}")

    print("\n--- Palindrome Check ---")
    print(f"Racecar: {processor.is_palindrome('Racecar')}")
    print(f"A man, a plan, a canal: Panama: {processor.is_palindrome('A man, a plan, a canal: Panama')}")

    print("\n--- Caesar Cipher ---")
    print(f"Encrypt 'abc' with shift 3: {processor.caesar_cipher('abc', 3)}")
    print(f"Decrypt 'khoor' with shift -3: {processor.caesar_cipher('khoor', -3)}")

    print("\n--- Most Common Word ---")
    print(f"Text: 'the quick brown fox the fox' -> {processor.most_common_word('the quick brown fox the fox')}")

import pytest



@pytest.fixture
def processor():
    """Fixture to provide a fresh StringProcessor instance for each test."""
    return StringProcessor()

# 1. Test reverse_words
def test_reverse_words_basic(processor: StringProcessor):
    """Tests basic word reversal and space collapsing."""
    s = "hello world this is a test"
    expected = "test a is this world hello"
    assert processor.reverse_words(s) == expected

def test_reverse_words_extra_spaces(processor: StringProcessor):
    """Tests handling of multiple spaces between words."""
    s = "  leading   and   trailing  "
    expected = "trailing and leading"
    assert processor.reverse_words(s) == expected

# 2. Test count_vowels
def test_count_vowels_case_insensitivity(processor: StringProcessor):
    """Tests vowel counting ignoring case."""
    s = "AEIOUaeiou"
    assert processor.count_vowels(s) == 10

def test_count_vowels_mixed_string(processor: StringProcessor):
    """Tests vowel counting in a mixed string."""
    s = "Rhythm"  # No vowels
    assert processor.count_vowels(s) == 0
    s_2 = "Apple"
    assert processor.count_vowels(s_2) == 2

# 3. Test is_palindrome
def test_is_palindrome_true(processor: StringProcessor):
    """Tests a classic palindrome."""
    assert processor.is_palindrome("Racecar") == True

def test_is_palindrome_complex(processor: StringProcessor):
    """Tests a palindrome ignoring case, spaces, and punctuation."""
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

def test_is_palindrome_false(processor: StringProcessor):
    """Tests a non-palindrome string."""
    assert processor.is_palindrome("hello world") == False

# 4. Test caesar_cipher
def test_caesar_cipher_positive_shift(processor: StringProcessor):
    """Tests a standard positive shift."""
    assert processor.caesar_cipher("abc", 3) == "def"
    assert processor.caesar_cipher("XYZ", 3) == "ABC" # Wraps around

def test_caesar_cipher_negative_shift(processor: StringProcessor):
    """Tests a negative shift (decryption)."""
    assert processor.caesar_cipher("def", -3) == "abc"
    assert processor.caesar_cipher("A", -1) == "Z" # Wraps around

def test_caesar_cipher_non_alpha_chars(processor: StringProcessor):
    """Tests that non-alphabetic characters are preserved."""
    assert processor.caesar_cipher("Hello World!", 3) == "Khoor Zruog!"

# 5. Test most_common_word
def test_most_common_word_basic(processor: StringProcessor):
    """Tests finding the most common word."""
    s = "apple banana apple orange banana apple"
    assert processor.most_common_word(s) == "apple"

def test_most_common_word_tiebreaker(processor: StringProcessor):
    """Tests the tiebreaker rule (first word encountered wins)."""
    # 'the' appears first, even though 'a' appears later and also twice
    s = "the a the a" 
    assert processor.most_common_word(s) == "the"

def test_most_common_word_empty_string(processor: StringProcessor):
    """Tests behavior with an empty string."""
    assert processor.most_common_word("") is None

def test_most_common_word_no_words(processor: StringProcessor):
    """Tests behavior with only punctuation/spaces."""
    assert processor.most_common_word("!@# $ %") is None