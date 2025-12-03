import regex as re
from tests.common import gpt2_bytes_to_unicode
import json


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.b_to_i = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Loads vocab and merges from files"""

        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # De-serialize merges
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges
            ]

        # De-serialize vocab
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text):
        """Encodes a string into a sequence of tokens"""

        # GPT-2 pretokenization pattern
        pretok_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        encoded_str = []

        # Handle special tokens by splitting first
        if self.special_tokens:
            # Build pattern to match special tokens, sorted by length descending to handle overlaps
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
            pattern = f"({special_pattern})"
            
            # Split by special tokens
            parts = re.split(pattern, text)
            
            for part in parts:
                if not part:  # Skip empty strings from split
                    continue
                    
                if part in self.special_tokens:
                    # Handle special token
                    w_b = part.encode("utf-8")
                    encoded_str.append(self.b_to_i[w_b])
                else:
                    # Apply GPT-2 pretokenization to non-special parts
                    pretokenized = re.findall(pretok_pat, part)
                    for word in pretokenized:
                        # Convert to bytes
                        w_b = word.encode("utf-8")
                        # Convert bytes to a sequence of bytes wrapped in a list
                        w_b = [bytes([c]) for c in w_b]  # b'the' -> [b't', b'h', b'e']

                        # Apply merges
                        w_b = self._apply_merges(w_b)

                        # Encode final merges to ints
                        encoded_str.extend([self.b_to_i[b] for b in w_b])
        else:
            # No special tokens, just apply GPT-2 pretokenization
            pretokenized = re.findall(pretok_pat, text)
            for word in pretokenized:
                # Convert to bytes
                w_b = word.encode("utf-8")
                # Convert bytes to a sequence of bytes wrapped in a list
                w_b = [bytes([c]) for c in w_b]  # b'the' -> [b't', b'h', b'e']

                # Apply merges
                w_b = self._apply_merges(w_b)

                # Encode final merges to ints
                encoded_str.extend([self.b_to_i[b] for b in w_b])
        
        return encoded_str

    def _apply_merges(self, w_b):
        """Applies merges"""
        for merge in self.merges:
            i = 0
            while i < len(w_b) - 1:
                if (w_b[i], w_b[i+1]) == merge:
                    w_b = w_b[:i] + [w_b[i] + w_b[i+1]] + w_b[i+2:]
                else:
                    i += 1
        return w_b
        
    def encode_iterable(self, iterable):
        """Encodes from an iterable"""

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        """Decodes a sequence of tokens to a string"""

        decoded_bytes = [self.vocab[id] for id in ids]  # Map from integers to bytes
        catted_bytes = b"".join(decoded_bytes)

        # Decode into UTF-8 codec
        decoded_str = catted_bytes.decode("utf-8", errors="replace")
        return decoded_str


if __name__ == "__main__":
    vocab_path = "gpt2_vocab.json"
    merges_path = "gpt2_merges.txt"

    Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
