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
            The string with words reversed and normalized spacing.
        """
        words = s.split()
        return " ".join(words[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the total number of vowels (a, e, i, o, u) in a string, case-insensitive.

        Args:
            s: The input string.

        Returns:
            The total count of vowels.
        """
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

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
        Applies the Caesar cipher to alphabetic characters in a string.
        Only letters (a-z, A-Z) are shifted. Supports positive and negative shifts.

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
                # Keep non-alphabetic characters unchanged
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most frequently occurring word in a string (case-insensitive).
        If there is a tie, the word that appears first in the text is returned.

        Args:
            s: The input string.

        Returns:
            The most common word (lowercase), or None if the string is empty or contains no words.
        """
        # Use regex to find all sequences of word characters
        words = re.findall(r'\b\w+\b', s.lower())
        
        if not words:
            return None

        word_counts = Counter(words)
        max_count = 0
        most_common = None
        
        # Iterate through the original sequence of words to handle ties correctly
        seen_words = {}
        for word in words:
            count = word_counts[word]
            if count > max_count:
                max_count = count
                most_common = word
            # If count == max_count, we do nothing, preserving the first one found
            
        return most_common

# Example Usage (optional, for demonstration)
if __name__ == '__main__':
    processor = StringProcessor()
    
    print("--- Reverse Words ---")
    print(f"Input: '  hello world  test ' -> Output: '{processor.reverse_words('  hello world  test ')}'")

    print("\n--- Count Vowels ---")
    print(f"Input: 'Programming' -> Output: {processor.count_vowels('Programming')}")

    print("\n--- Is Palindrome ---")
    print(f"Input: 'A man, a plan, a canal: Panama' -> Output: {processor.is_palindrome('A man, a plan, a canal: Panama')}")
    print(f"Input: 'hello' -> Output: {processor.is_palindrome('hello')}")

    print("\n--- Caesar Cipher ---")
    print(f"Input: 'abc xyz' Shift 3 -> Output: '{processor.caesar_cipher('abc xyz', 3)}'")
    print(f"Input: 'xyz' Shift -3 -> Output: '{processor.caesar_cipher('xyz', -3)}'")

    print("\n--- Most Common Word ---")
    text = "The quick brown fox jumps over the lazy fox"
    print(f"Input: '{text}' -> Output: {processor.most_common_word(text)}") # Should be 'the' (appears twice, first)
    text_tie = "apple banana apple banana"
    print(f"Input: '{text_tie}' -> Output: {processor.most_common_word(text_tie)}") # Should be 'apple' (first encountered)

import pytest



@pytest.fixture
def processor() -> StringProcessor:
    """Fixture to provide a fresh StringProcessor instance for each test."""
    return StringProcessor()

# 1. reverse_words tests
def test_reverse_words_basic(processor: StringProcessor):
    assert processor.reverse_words("hello world") == "world hello"

def test_reverse_words_with_extra_spaces(processor: StringProcessor):
    assert processor.reverse_words("  leading and trailing  ") == "trailing and leading"

def test_reverse_words_empty_string(processor: StringProcessor):
    assert processor.reverse_words("") == ""

# 2. count_vowels tests
def test_count_vowels_mixed_case(processor: StringProcessor):
    assert processor.count_vowels("AEIOUaeiou") == 10

def test_count_vowels_no_vowels(processor: StringProcessor):
    assert processor.count_vowels("rhythm dry") == 0

# 3. is_palindrome tests
def test_is_palindrome_true_complex(processor: StringProcessor):
    # Classic palindrome test
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

def test_is_palindrome_false(processor: StringProcessor):
    assert processor.is_palindrome("hello world") == False

def test_is_palindrome_simple_true(processor: StringProcessor):
    assert processor.is_palindrome("Racecar") == True

# 4. caesar_cipher tests
def test_caesar_cipher_positive_shift(processor: StringProcessor):
    # Shift by 3 (A -> D)
    assert processor.caesar_cipher("ABC", 3) == "DEF"
    # Shift by 3 (z -> c, wrapping)
    assert processor.caesar_cipher("xyz", 3) == "abc"

def test_caesar_cipher_negative_shift(processor: StringProcessor):
    # Shift by -1 (B -> A)
    assert processor.caesar_cipher("BCD", -1) == "ABC"
    # Shift by -3 (c -> z, wrapping)
    assert processor.caesar_cipher("abc", -3) == "xyz"

def test_caesar_cipher_non_alpha_chars(processor: StringProcessor):
    # Should ignore spaces and punctuation
    assert processor.caesar_cipher("Hello World!", 1) == "Ifmmp Xpsme!"

# 5. most_common_word tests
def test_most_common_word_basic(processor: StringProcessor):
    text = "apple banana apple orange"
    # 'apple' appears twice, 'banana' once
    assert processor.most_common_word(text) == "apple"

def test_most_common_word_tie_first_wins(processor: StringProcessor):
    # Both 'the' and 'a' appear twice. 'the' appears first.
    text = "The a the a"
    assert processor.most_common_word(text) == "the"

def test_most_common_word_empty_string(processor: StringProcessor):
    assert processor.most_common_word("") is None

def test_most_common_word_only_punctuation(processor: StringProcessor):
    assert processor.most_common_word("!@#$%^") is None

def test_most_common_word_case_insensitivity(processor: StringProcessor):
    text = "Word word WORD"
    assert processor.most_common_word(text) == "word"