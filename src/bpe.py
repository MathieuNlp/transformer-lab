from collections import Counter, deque
from functools import lru_cache
import json
from multiprocessing import process

class Tokenizer:
    def __init__(self):
        # Map id -> str
        self.vocab = {}
        # Map str -> id
        self.inverse_vocab = {}
        # BPE merges (str1, str2) -> str_merged$
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    def train(self, text: str, vocab_size: int, allowed_special: set={"<|endoftext|>"}):
        # Preprocess the data -> Replace space with "Ġ"
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)
        
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text))
            if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}




if __name__ == "__main__":
    tokenizer = Tokenizer()
    text = "This chapter is about tokenization."
    VOCAB_SIZE = 52_000
    tokenizer.train(text, VOCAB_SIZE)
    print(tokenizer.vocab[196])