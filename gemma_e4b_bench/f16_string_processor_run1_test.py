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
            The string with words in reverse order and single spaces between them.
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
        # Tokenize: find all sequences of letters and convert to lowercase
        words = re.findall(r'[a-zA-Z]+', s.lower())
        
        if not words:
            return None

        word_counts = Counter(words)
        max_count = 0
        most_common = None
        
        # Iterate through the original sequence of words to handle ties based on first appearance
        seen_words = {}
        for word in words:
            if word_counts[word] > max_count:
                max_count = word_counts[word]
                most_common = word
            # If counts are equal, we keep the current 'most_common' because it appeared earlier
            
        return most_common

# Example Usage (optional, for testing environment setup)
if __name__ == '__main__':
    processor = StringProcessor()
    
    print("--- Reverse Words ---")
    print(f"Input: 'Hello world   this is a test'")
    print(f"Output: '{processor.reverse_words('Hello world   this is a test')}'")

    print("\n--- Count Vowels ---")
    print(f"Input: 'Programming is fun'")
    print(f"Output: {processor.count_vowels('Programming is fun')}")

    print("\n--- Is Palindrome ---")
    print(f"Input: 'A man, a plan, a canal: Panama'")
    print(f"Output: {processor.is_palindrome('A man, a plan, a canal: Panama')}")
    
    print("\n--- Caesar Cipher ---")
    print(f"Input: 'ABC xyz', Shift: 3")
    print(f"Output: '{processor.caesar_cipher('ABC xyz', 3)}'")
    print(f"Input: 'xyz', Shift: -1")
    print(f"Output: '{processor.caesar_cipher('xyz', -1)}'")

    print("\n--- Most Common Word ---")
    print(f"Input: 'the quick brown fox jumps over the lazy fox'")
    print(f"Output: {processor.most_common_word('the quick brown fox jumps over the lazy fox')}")

import pytest



@pytest.fixture
def processor() -> StringProcessor:
    """Fixture to provide a fresh StringProcessor instance for each test."""
    return StringProcessor()

# --- Tests for reverse_words ---
def test_reverse_words_standard(processor: StringProcessor):
    """Tests basic word reversal and space collapsing."""
    s = "one two three four"
    expected = "four three two one"
    assert processor.reverse_words(s) == expected

def test_reverse_words_multiple_spaces(processor: StringProcessor):
    """Tests handling of excessive whitespace."""
    s = "  leading   and   trailing  "
    expected = "trailing and leading"
    assert processor.reverse_words(s) == expected

# --- Tests for count_vowels ---
def test_count_vowels_mixed_case(processor: StringProcessor):
    """Tests vowel counting with mixed casing."""
    s = "AEIOUaeiou"
    assert processor.count_vowels(s) == 10

def test_count_vowels_no_vowels(processor: StringProcessor):
    """Tests a string containing no vowels."""
    s = "rhythm dry fly"
    assert processor.count_vowels(s) == 0

# --- Tests for is_palindrome ---
def test_is_palindrome_true(processor: StringProcessor):
    """Tests a classic palindrome with punctuation and spaces."""
    s = "A man, a plan, a canal: Panama"
    assert processor.is_palindrome(s) == True

def test_is_palindrome_false(processor: StringProcessor):
    """Tests a non-palindrome string."""
    s = "hello world"
    assert processor.is_palindrome(s) == False

# --- Tests for caesar_cipher ---
def test_caesar_cipher_positive_shift(processor: StringProcessor):
    """Tests a standard positive shift (Caesar cipher)."""
    s = "abc"
    shift = 3
    expected = "def"
    assert processor.caesar_cipher(s, shift) == expected

def test_caesar_cipher_negative_shift_wrap(processor: StringProcessor):
    """Tests a negative shift that wraps around the alphabet."""
    s = "abc"
    shift = -1
    expected = "zab"
    assert processor.caesar_cipher(s, shift) == expected

# --- Tests for most_common_word ---
def test_most_common_word_basic(processor: StringProcessor):
    """Tests finding the clear most common word."""
    s = "apple banana apple orange banana apple"
    # 'apple' appears 3 times, 'banana' 2 times
    assert processor.most_common_word(s) == "apple"

def test_most_common_word_tie_breaker(processor: StringProcessor):
    """Tests the tie-breaker rule (first appearance wins)."""
    s = "cat dog cat dog"
    # Both 'cat' and 'dog' appear twice. 'cat' appears first.
    assert processor.most_common_word(s) == "cat"

def test_most_common_word_empty_input(processor: StringProcessor):
    """Tests behavior with an empty string."""
    assert processor.most_common_word("") is None

def test_most_common_word_only_punctuation(processor: StringProcessor):
    """Tests behavior when the string contains no actual words."""
    assert processor.most_common_word("!@#$%^&*()") is None