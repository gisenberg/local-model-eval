

import re

class StringProcessor:
    """
    A utility class for performing various string manipulations and analyses.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string and collapses multiple spaces into a single space.

        Args:
            s: The input string.

        Returns:
            A string with the word order reversed and single spaces between words.
        """
        if not s:
            return ""
        # Split by whitespace, filter out empty strings, reverse, and join with a single space
        words = s.split()
        return " ".join(words[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the total number of vowels (a, e, i, o, u) in a string, case-insensitively.

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
        # 1. Convert to lowercase
        s_lower = s.lower()
        # 2. Remove non-alphanumeric characters (spaces, punctuation, etc.)
        cleaned_s = re.sub(r'[^a-z0-9]', '', s_lower)
        
        # 3. Check if it reads the same forwards and backwards
        return cleaned_s == cleaned_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher shift to alphabetic characters in a string.
        Non-alphabetic characters are left unchanged. Supports negative shifts.

        Args:
            s: The input string.
            shift: The integer shift value (positive or negative).

        Returns:
            The encrypted or decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Handle lowercase letters
                start = ord('a')
                new_ord = (ord(char) - start + shift) % 26
                result.append(chr(start + new_ord))
            elif 'A' <= char <= 'Z':
                # Handle uppercase letters
                start = ord('A')
                new_ord = (ord(char) - start + shift) % 26
                result.append(chr(start + new_ord))
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
            The most common word (in its original casing from the first appearance), 
            or None if the string is empty or contains no words.
        """
        # Tokenize: find all sequences of letters
        words = re.findall(r'[a-zA-Z]+', s)
        if not words:
            return None

        # Use a dictionary to track counts and first appearance index
        word_counts = {}
        word_order = []
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            
            if lower_word not in word_counts:
                word_counts[lower_word] = 0
                # Store the original casing of the first appearance
                word_counts[lower_word] = (0, word) 
                word_order.append(lower_word)
            
            # Increment count
            count, _ = word_counts[lower_word]
            word_counts[lower_word] = (count + 1, word_counts[lower_word][1])

        # Find the maximum count
        max_count = 0
        for count, _ in word_counts.values():
            max_count = max(max_count, count)

        # Iterate through the order of appearance to find the first word matching max_count
        for lower_word in word_order:
            count, original_casing = word_counts[lower_word]
            if count == max_count:
                return original_casing
        
        return None # Should not happen if words list was not empty


if __name__ == '__main__':
    # Example Usage
    processor = StringProcessor()
    
    print("--- Reverse Words ---")
    print(f"Original: '  hello world   test' -> Reversed: '{processor.reverse_words('  hello world   test')}'")

    print("\n--- Count Vowels ---")
    print(f"String: 'Programming' -> Vowels: {processor.count_vowels('Programming')}")

    print("\n--- Is Palindrome ---")
    print(f"Palindrome 'Racecar': {processor.is_palindrome('Racecar')}")
    print(f"Palindrome 'A man, a plan, a canal: Panama': {processor.is_palindrome('A man, a plan, a canal: Panama')}")

    print("\n--- Caesar Cipher ---")
    print(f"Encrypt 'Hello' with shift 3: '{processor.caesar_cipher('Hello', 3)}'")
    print(f"Decrypt 'Khoor' with shift -3: '{processor.caesar_cipher('Khoor', -3)}'")
    
    print("\n--- Most Common Word ---")
    text = "The quick brown fox jumps over the lazy fox. The fox is quick."
    print(f"Text: '{text}' -> Most Common: {processor.most_common_word(text)}")

import pytest


@pytest.fixture
def processor():
    """Fixture to provide a fresh StringProcessor instance for each test."""
    return StringProcessor()

# --- Test Cases for reverse_words ---
def test_reverse_words_basic(processor):
    """Tests basic word reversal and space collapsing."""
    assert processor.reverse_words("one two three") == "three two one"

def test_reverse_words_with_extra_spaces(processor):
    """Tests handling of leading, trailing, and multiple internal spaces."""
    input_str = "  leading   middle trailing  "
    expected = "trailing middle leading"
    assert processor.reverse_words(input_str) == expected

def test_reverse_words_empty_string(processor):
    """Tests behavior with an empty input string."""
    assert processor.reverse_words("") == ""

# --- Test Cases for count_vowels ---
def test_count_vowels_mixed_case(processor):
    """Tests vowel counting with mixed case."""
    assert processor.count_vowels("AEIOUaeiou") == 10

def test_count_vowels_no_vowels(processor):
    """Tests counting vowels in a string with no vowels."""
    assert processor.count_vowels("rhythm fly") == 0

# --- Test Cases for is_palindrome ---
def test_is_palindrome_true(processor):
    """Tests a standard palindrome with punctuation and mixed case."""
    assert processor.is_palindrome("Madam, I'm Adam") == True

def test_is_palindrome_false(processor):
    """Tests a non-palindrome string."""
    assert processor.is_palindrome("hello world") == False

# --- Test Cases for caesar_cipher ---
def test_caesar_cipher_positive_shift(processor):
    """Tests encryption with a positive shift."""
    # H -> K (+3)
    assert processor.caesar_cipher("Hi", 3) == "Kl"

def test_caesar_cipher_negative_shift(processor):
    """Tests decryption (negative shift) and wrapping."""
    # K -> H (-3)
    assert processor.caesar_cipher("Khoor", -3) == "Hello"

# --- Test Cases for most_common_word ---
def test_most_common_word_basic(processor):
    """Tests finding the most common word."""
    text = "apple banana apple cherry banana apple"
    # 'apple' appears 3 times, 'banana' appears 2 times
    assert processor.most_common_word(text) == "apple"

def test_most_common_word_tie_breaker(processor):
    """Tests the tie-breaker rule (first word encountered wins)."""
    # 'the' appears first, even though 'a' might appear later if we didn't track order
    text = "the a the a" 
    # 'the' appears first (index 0) and has count 2. 'a' appears second (index 1) and has count 2.
    # The implementation should return the one that appeared first in the text.
    assert processor.most_common_word(text) == "the"

def test_most_common_word_empty_input(processor):
    """Tests behavior with an empty string."""
    assert processor.most_common_word("") is None

def test_most_common_word_only_punctuation(processor):
    """Tests behavior when the string contains no letters."""
    assert processor.most_common_word("!@#$%^&*()") is None