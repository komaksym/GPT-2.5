import regex as re
from collections import defaultdict


special_tokens = ["<|endoftext|>", "<start>", "<end>"]

vocab_size = 268
num_of_merges = vocab_size - 256


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_data(input_path):
    with open(input_path, "r") as f:
        corpus = f.read()
    return corpus


def strip_of_special_tokens(corpus, special_tokens):
    """Strips of special tokens to avoid counting them as bytes"""

    # Escape | delimiter in special tokens
    for i in range(len(special_tokens)):
        if "|" in special_tokens[i]:
            special_tokens[i] = re.escape(special_tokens[i])

    # Join special tokens into a delim for a splitting pattern
    delim = "|".join(special_tokens)
    chunks = re.split(delim, corpus)
    # Remove empty chunks
    chunks = [ch for ch in chunks if ch.strip()]
    return chunks


# Pre-tokenization
def pretokenize(corpus, ptrn):
    """Pre-tokenizes on regex pattern"""

    counts = {}
    for t in corpus:
        for word in re.findall(ptrn, t):
            counts[word] = counts.get(word, 0) + 1
    return counts


def split_to_bytes(corpus):
    """Splits words by characters and counts frequency"""

    counts = {}
    # Count byte pairs
    for k, v in corpus.items():
        new_key = tuple([c for c in k])
        counts[new_key] = v

    # Sort by the highest frequency
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return counts


def count_bytepairs(corpus, bp_to_counts=None, bp_to_words=None, mf_pair=None, merged_words=None):
    """Counts bytepair frequencies in the corpus
    If ran the first time (no counts provided) then count all byte pairs in the whole corpus
    If ran consecutively, remove the most frequent pair count as it's merged now
    and count byte pairs only on the merged words now"""

    # If counts are provided, update only the affected byte pairs
    if bp_to_counts and bp_to_words:
        # Remove the merged pair itself
        bp_to_counts.pop(mf_pair, None)
        bp_to_words.pop(mf_pair, None)

        # Clean up word references: remove words that no longer exist in corpus
        # and update counts for pairs that lost words
        for pair in list(bp_to_words.keys()):
            old_words = bp_to_words[pair]
            # Find words that are no longer in corpus (they were merged)
            removed_words = {w for w in old_words if w not in corpus}

            if removed_words:
                # Update the word set
                bp_to_words[pair] = old_words - removed_words

                # Recalculate count for this pair from remaining words
                bp_to_counts[pair] = sum(corpus[w] for w in bp_to_words[pair])

                # Remove empty sets
                if not bp_to_words[pair]:
                    bp_to_words.pop(pair)
                    bp_to_counts.pop(pair, None)

        # Add counts for new byte pairs in merged words
        for w in merged_words:
            for c1, c2 in zip(w, w[1:]):
                bp_to_counts[(c1, c2)] = bp_to_counts.get((c1, c2), 0) + corpus[w]
                bp_to_words[(c1, c2)].add(w)

    # For the first time we need to count every single pair
    else:
        bp_to_counts = {}
        bp_to_words = defaultdict(set)

        # Count every single pair in the corpus
        for k, v in corpus.items():
            for c1, c2 in zip(k, k[1:]):
                bp_to_counts[(c1, c2)] = bp_to_counts.get((c1, c2), 0) + v
                bp_to_words[(c1, c2)].add(k)

    return bp_to_counts, bp_to_words


def get_mf_pair(counts, counts_to_words):
    """Takes the most frequent byte pair along with words with that pair and returns them"""

    # Get the max frequency
    maxf = counts[max(counts, key=counts.get)]

    # Get the candidates with the max frequency
    candidates = [k for k, v in counts.items() if v == maxf]
    # Pick the lexicographically greater pair
    pair = max(candidates)
    return pair, counts_to_words[pair]


def merge(corpus, merge_pair, merge_pair_words):
    """Merges the word in the corpus
    by joining two bytes into one and
    re-assigning the key in the dictionary"""

    # Keeps track of merged words to update byte counting only on these as these only change
    merged_words = set()

    # Merging the keys
    for w in merge_pair_words:
        new_k = []
        b = 0
        while b < len(w):
            if b + 1 < len(w):
                c1, c2 = w[b], w[b + 1]
                if c1 + c2 == merge_pair:
                    new_k.append(merge_pair)
                    b += 2
                else:
                    new_k.append(c1)
                    b += 1
            else:
                new_k.append(w[b])
                b += 1
        # Add new merged word
        corpus[tuple(new_k)] = corpus[w]

        # Add to merged words to optimize byte pair counts as this only changes and the rest is still the same
        merged_words.add(tuple(new_k))

    # Pop the unmerged words
    for w in merge_pair_words:
        corpus.pop(w)

    return corpus, merged_words


def pair_to_bytes(pair):
    return tuple(b.encode("utf-8") for b in pair)


def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {}
    merges = []
    mf_pair = None
    merged_words = None
    num_of_merges = vocab_size - 256

    corpus = read_data(input_path)
    corpus = strip_of_special_tokens(corpus, special_tokens)
    corpus = pretokenize(corpus, PAT)
    corpus = split_to_bytes(corpus)

    # Start merges
    for i in range(num_of_merges):
        if i == 0:
            # Count bytepairs
            counts, counts_to_words = count_bytepairs(corpus)
        else:
            counts, counts_to_words = count_bytepairs(
                corpus, counts, counts_to_words, mf_pair, merged_words
            )
            # Check if there are any pairs left to merge
            if not counts:
                # No more pairs to merge, stop early
                break
            # Get the most frequent pair
            mf_pair, mf_pair_words = get_mf_pair(counts, counts_to_words)
            # Add merge to merges
            pair_b = pair_to_bytes(mf_pair)
            merges.append(pair_b)
            # Add the merge to the vocab
            merge_b = "".join(mf_pair).encode("utf-8")
            vocab[256 + i] = merge_b
            # Apply the merge to the corpus
            corpus, merged_words = merge(corpus, "".join(mf_pair), mf_pair_words)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 262, special_tokens = ["<|endoftext|>", "<start>", "<end>"])
