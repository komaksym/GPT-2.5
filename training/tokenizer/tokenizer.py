import regex as re
from .common import gpt2_bytes_to_unicode
import json
import os
import typing
from collections.abc import Iterable, Generator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: typing.Optional[list[str]] = None,
    ):
        """
        Initializes the Tokenizer with a vocabulary and BPE merges.
        vocab: Map from token ID to bytes.
        merges: List of byte pairs to merge.
        special_tokens: Optional list of special strings to treat as single tokens.
        """
        self.vocab = vocab
        self.b_to_i = {v: k for k, v in self.vocab.items()}  # Bytes to ints
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: typing.Optional[list[str]] = None,
    ) -> "Tokenizer":
        """
        Loads vocabulary and BPE merges from disk (standard GPT-2 format).
        """

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

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a sequence of token IDs.
        Handles special tokens first, then applies GPT-2 pre-tokenization and BPE merges.
        """

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
                    
                elif part in self.special_tokens:
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

    def _apply_merges(self, w_b: list[bytes]) -> list[bytes]:
        """
        Sequentially applies BPE merges to a list of bytes.
        """

        for merge in self.merges:
            i = 0
            while i < len(w_b) - 1:
                if (w_b[i], w_b[i+1]) == merge: # If a merge was found
                    # Merge the two matching bytes 
                    # and keep everything like is before and after the matching merge pair
                    w_b = w_b[:i] + [w_b[i] + w_b[i+1]] + w_b[i+2:]
                else: # Else keep searching for merges
                    i += 1
        return w_b
        
    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int, None, None]:
        """
        Encodes a sequence of strings, yielding token IDs one by one.
        """

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.
        """

        decoded_bytes = [self.vocab[id] for id in ids]  # Map from integers to bytes
        catted_bytes = b"".join(decoded_bytes)

        # Decode into UTF-8 codec
        decoded_str = catted_bytes.decode("utf-8", errors="replace")
        return decoded_str


if __name__ == "__main__":
    vocab_path = "gpt2_vocab.json"
    merges_path = "gpt2_merges.txt"

    Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
