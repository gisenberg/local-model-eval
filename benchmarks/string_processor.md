# Benchmark: String Processor

**Difficulty:** Easy
**Expected tests:** 5
**Skills tested:** String manipulation, regex, basic data structures, test writing

## Prompt

```
Write a Python class called StringProcessor with the following methods:

1. reverse_words(s: str) -> str — reverse the order of words in a string (not the characters). Multiple spaces between words should become a single space. Leading/trailing spaces removed.
2. count_vowels(s: str) -> int — count vowels (a, e, i, o, u, case-insensitive) in the string
3. is_palindrome(s: str) -> bool — check if the string is a palindrome, ignoring case, spaces, and punctuation
4. caesar_cipher(s: str, shift: int) -> str — apply Caesar cipher with given shift. Only shift a-z and A-Z, leave other characters unchanged. Support negative shifts.
5. most_common_word(s: str) -> Optional[str] — return the most frequently occurring word (case-insensitive). If tied, return the one that appears first. Return None for empty strings.

Include type hints and a brief docstring on each method.
Write 5 pytest tests covering each method.
```

## What Makes This a Good Baseline

- Every task is a well-known string operation — no algorithmic complexity
- Tests whether the model can handle basic prompt-following and test-writing
- Establishes a quality floor — any model that fails this has fundamental issues
- Fast to generate and test (small code output)

## Evaluation Criteria

- Does the code import without errors?
- Do all 5 tests pass when run with `pytest -v`?
- No manual fixes allowed
