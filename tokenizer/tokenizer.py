import regex as re
from common import gpt2_bytes_to_unicode
from train_tokenizer import strip_of_special_tokens
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

        pretok_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Pretokenize
        #pretokenized = re.findall(pretok_pat, text[0] if isinstance(text, list) else text)
        pretokenized = re.split(r'\s', text)
        encoded_str = []

        # Encode each word independently
        for word in pretokenized:
            # Convert the w_bord to bytes first
            w_b = word.encode("utf-8")
            # Convert bytes to a sequence of bytes wrapped in a list
            w_b = [bytes([c]) for c in w_b]  # b'the' -> [b't', b'h', b'e']

            # Start merges
            for merge in self.merges:
                for idx, (b1, b2) in enumerate(zip(w_b, w_b[1:])):
                    # If a match was found
                    if merge == (b1, b2):
                        # Merge the bytes
                        w_b = w_b[:idx] + [b1 + b2] + w_b[idx + 2 :]
                        # Breakout to update the merged string
                        break

            # Encode final merges to ints
            encoded_str.extend([self.b_to_i[b] for b in w_b])
        

        return encoded_str

    def encode_iterable(self, iterable):
        """Encodes from an iterable"""

        for text in iterable:
            yield(self.encode(text))

    def decode(self, ids):
        """Decodes a sequence of tokens to a string"""

        decoded_bytes = b"".join([self.vocab[id] for id in ids])  # Map from integers to bytes
        # Decode into UTF-8 codec
        decoded_str = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_str


if __name__ == "__main__":
    vocab_path = "gpt2_vocab.json"
    merges_path = "gpt2_merges.txt"

    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    encoded_ids = tok.encode(text)
    decoded = tok.decode(encoded_ids)
    print(decoded)

