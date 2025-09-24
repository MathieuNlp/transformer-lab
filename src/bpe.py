from collections import Counter, deque
from functools import lru_cache
import json
from multiprocessing import process

class BpeAlgo:
    def __init__(self):
        # id -> str
        self.vocab = {}
        # str -> id
        self.inverse_vocab = {}
        #(token id1, token id2) -> merged token id
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    def train(self, text: str, vocab_size: int, allowed_special: set={"<|endoftext|>"}):
        # Preprocess text -> Replace space with "Ġ"
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)
        
        # Take the first 256 characters of utf-8 (ASCII) to put on the vocab
        unique_chars = [chr(i) for i in range(256)]
        # Add following to the 256 first characters, the characters of the processed text
        unique_chars.extend(
            char for char in sorted(set(processed_text))
            if char not in unique_chars
        )
        # Add the space character to the vocab at the end
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        # Load the vocab and inverse vocab mappings
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add special tokens if needed at the end of the initialised vocab
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    token_id = len(self.vocab)
                    self.vocab[token_id] = token
                    self.inverse_vocab[token] = token_id
        
        # Tokenize the processed text
        token_ids = [self.inverse_vocab[char] for char in processed_text]
        # BPE algorithm
        for new_id in range(len(self.vocab), vocab_size):
        # 1) Find the pair of token_id with most frequency
            # Output the most frequent pair of token id
            pair_id = BpeAlgo.find_freq_pairs(token_ids)
            if pair_id is None:
                break
        # 2) Merge the pair of tokens
            self.bpe_merges[pair_id] = new_id
            print(self.bpe_merges)
        # 3) Merge the 2 tokens in the tokenizer text
            token_ids = BpeAlgo.replace_pair(token_ids, pair_id, new_id)
        # 4) Update vocab
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    @staticmethod
    def find_freq_pairs(token_ids: list[int]) -> tuple[int, int]:
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        
        return max(pairs.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []
        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)

        return replaced

if __name__ == "__main__":
    tokenizer = BpeAlgo()
    text = "This chapter is about tokenization and we are absolutely delighted to learn about it has it is a core component of ai today."
    VOCAB_SIZE = 52_000
    tokenizer.train(text, VOCAB_SIZE)
