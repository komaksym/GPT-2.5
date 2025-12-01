import regex as re
from .common import gpt2_bytes_to_unicode
import json


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.b_to_i = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # Merges
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges
            ]

        # Vocab
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text):
        """Encodes a string into a sequence of tokens"""

        pretok_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokenized = re.findall(pretok_pat, text)
        encoded_str = []

        # Encode each word independently
        for w in pretokenized:
            # Convert a word to a sequence of bytes wrapped in a list
            w = [bytes([ord(c)]) for c in w]  # 'the' -> [b't', b'h', b'e']

            # Start merges
            for merge in self.merges:
                for idx, (b1, b2) in enumerate(zip(w, w[1:])):
                    # If a match was found
                    if merge == (b1, b2):
                        # Merge the bytes
                        w = [b1 + b2] + w[idx + 2 :]
                        # Breakout to update the merged string
                        break

            # Encode final merges to ints
            encoded_str.extend([self.b_to_i[b] for b in w])

        return encoded_str

    def encode_iterable(self, iterable):
        pass

    def decode(self, ids):
        """Decodes a sequence of tokens to a string"""

        decoded_bytes = [self.vocab[id] for id in ids]  # Map from integers to bytes
        decoded_str = ""

        # Decode into UTF-8 codec
        for byte in decoded_bytes:
            s = byte.decode("utf-8", errors="replace")
            # Concat the strings
            decoded_str += s

        return decoded_str


if __name__ == "__main__":
    vocab_path = "gpt2_vocab.json"
    merges_path = "gpt2_merges.txt"

    Tokenizer.from_files(vocab_path, merges_path)
