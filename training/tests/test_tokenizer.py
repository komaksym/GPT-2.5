from tokenizer.tokenizer import Tokenizer


def _build_tokenizer(*, special_tokens: list[str] | None = None) -> Tokenizer:
    """Create a small tokenizer fixture with deterministic merges."""
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"abc",
        5: b"<end>",
        6: b"<endoftext>",
    }
    merges = [(b"a", b"b"), (b"ab", b"c")]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def test_encode_without_special_tokens_applies_bpe_merges():
    """Encode plain text with the configured byte-pair merges."""
    tokenizer = _build_tokenizer()

    token_ids = tokenizer.encode("abc")

    assert token_ids == [4]
    assert tokenizer.decode(token_ids) == "abc"


def test_encode_prefers_longest_overlapping_special_token():
    """Match longer special tokens before shorter overlapping ones."""
    tokenizer = _build_tokenizer(special_tokens=["<end>", "<endoftext>"])

    token_ids = tokenizer.encode("<endoftext><end>")

    assert token_ids == [6, 5]


def test_encode_mixes_special_tokens_with_regular_text_and_iterables():
    """Preserve ordering across regular text, special tokens, and iterables."""
    tokenizer = _build_tokenizer(special_tokens=["<end>"])

    token_ids = tokenizer.encode("abc<end>abc")

    assert token_ids == [4, 5, 4]
    assert list(tokenizer.encode_iterable(["abc", "<end>", "abc"])) == [4, 5, 4]
