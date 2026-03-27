import json
import os
import typing
from collections.abc import Iterable, Generator

import regex as re

from .common import gpt2_bytes_to_unicode

PRETOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRETOKENIZATION_REGEX = re.compile(PRETOKENIZATION_PATTERN)


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
        self.b_to_i = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self._special_tokens = tuple(special_tokens or ())
        self._special_token_set = set(self._special_tokens)
        self._special_token_splitter = self._build_special_token_splitter(
            self._special_tokens
        )

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
                gpt2_vocab_index: bytes(
                    [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
                )
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _build_special_token_splitter(
        special_tokens: tuple[str, ...],
    ) -> re.Pattern | None:
        """Compile the regex used to isolate configured special tokens."""
        if not special_tokens:
            return None

        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
        return re.compile(f"({special_pattern})")

    def _encode_text_chunk(self, text: str) -> list[int]:
        """Encode one non-special-text chunk with GPT-2 pretokenization and BPE."""
        encoded_tokens: list[int] = []
        for word in PRETOKENIZATION_REGEX.findall(text):
            word_bytes = [bytes([byte]) for byte in word.encode("utf-8")]
            merged_word = self._apply_merges(word_bytes)
            encoded_tokens.extend(
                self.b_to_i[token_bytes] for token_bytes in merged_word
            )
        return encoded_tokens

    def _iter_text_chunks(self, text: str) -> Iterable[str]:
        """Yield the input split into text and special-token chunks."""
        if self._special_token_splitter is None:
            yield text
            return

        for part in self._special_token_splitter.split(text):
            if part:
                yield part

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a sequence of token IDs.
        Handles special tokens first, then applies GPT-2 pre-tokenization and BPE merges.
        """
        encoded_tokens: list[int] = []
        for chunk in self._iter_text_chunks(text):
            if chunk in self._special_token_set:
                encoded_tokens.append(self.b_to_i[chunk.encode("utf-8")])
                continue
            encoded_tokens.extend(self._encode_text_chunk(chunk))
        return encoded_tokens

    def _apply_merges(self, w_b: list[bytes]) -> list[bytes]:
        """
        Sequentially applies BPE merges to a list of bytes.
        """

        for merge in self.merges:
            i = 0
            while i < len(w_b) - 1:
                if (w_b[i], w_b[i + 1]) == merge:
                    w_b = w_b[:i] + [w_b[i] + w_b[i + 1]] + w_b[i + 2 :]
                else:
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

        return catted_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    vocab_path = "gpt2_vocab.json"
    merges_path = "gpt2_merges.txt"

    Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
