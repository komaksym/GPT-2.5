import regex as re
from collections import defaultdict
from tqdm import tqdm
import json
from itertools import islice


special_tokens = ["<|endoftext|>", "<start>", "<end>"]

vocab_size = 268
num_of_merges = vocab_size - 256

pretok_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_data(input_path):
    with open(input_path) as f:
        corpus = f.read()
    return corpus


def strip_of_special_tokens(corpus, special_tokens):
    """Strips of special tokens to avoid counting them as bytes"""

    # Escape | delimiter in special tokens
    escaped_special_tokens = []
    for token in special_tokens:
        if "|" in token:
            escaped_special_tokens.append(re.escape(token))
        else:
            escaped_special_tokens.append(token)

    # Join special tokens into a delim for a splitting pattern
    delim = "|".join(escaped_special_tokens)
    chunks = re.split(delim, corpus)
    # Remove empty chunks
    chunks = [ch for ch in chunks if ch.strip()]
    return chunks


# Pre-tokenization
def pretokenize(corpus, ptrn):
    """Pre-tokenizes on regex pattern and counts words"""

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
        new_key = tuple([c for c in k]) # "iron" -> ('i', 'r', 'o', 'n')
        counts[new_key] = v

    # Sort by the highest frequency
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return counts


def count_bytepairs(corpus, bp_to_counts=None, bp_to_words=None, 
                    merged_words=None, removed_word_freqs=None):
    """Counts bytepair frequencies in the corpus
    If ran the first time (no counts provided) then count all byte pairs in the whole corpus
    If ran consecutively, update counts based on removed words and new merged words"""

    # If counts are provided, update only the affected byte pairs
    if bp_to_counts and bp_to_words:
        # 1. Decrement counts for pairs in the words that are being removed (merged)
        if removed_word_freqs:
            for w, freq in removed_word_freqs.items():
                # Find all pairs in w
                current_pairs = []
                for i in range(len(w) - 1):
                    current_pairs.append((w[i], w[i+1]))
                
                # Decrement counts
                for p in current_pairs:
                    if p in bp_to_counts:
                        bp_to_counts[p] -= freq
                        if bp_to_counts[p] <= 0:
                             if p in bp_to_counts: del bp_to_counts[p]
                
                # Remove w from bp_to_words for unique pairs
                for p in set(current_pairs):
                    if p in bp_to_words and w in bp_to_words[p]:
                        bp_to_words[p].remove(w)
                        if not bp_to_words[p]:
                            del bp_to_words[p]

        # 2. Add counts for new byte pairs in merged words
        for w in merged_words:
            freq = corpus[w]
            current_pairs = []
            for i in range(len(w) - 1):
                current_pairs.append((w[i], w[i+1]))
            
            for p in current_pairs:
                bp_to_counts[p] = bp_to_counts.get(p, 0) + freq
                bp_to_words[p].add(w)

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

        # Add to merged words to optimize byte pair counts
        # As this only changes and the rest is still the same
        merged_words.add(tuple(new_k))

    # Pop the unmerged words
    removed_word_freqs = {}
    for w in merge_pair_words:
        removed_word_freqs[w] = corpus.pop(w)

    return corpus, merged_words, removed_word_freqs


def pair_to_bytes(pair):
    """Converts a tuple of strings into a tuple of bytes"""

    return tuple(b.encode("utf-8") for b in pair) # ('a', 'c') -> (b'a', b'c')


def train_bpe(input_path, vocab_size, special_tokens):
    """Trains BPE on given corpus based on specified vocab size
        and returns vocab and merges"""

    # Fill the vocab with initial 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Some variables that will take part in the training process
    merges = []
    merged_words = None
    removed_word_freqs = None
    num_of_merges = vocab_size - 256 - len(special_tokens)

    # Reads the data
    corpus = read_data(input_path)[:1000]
    #corpus = read_data(input_path)
    # Strips of special tokens to avoid counting them in training process
    corpus = strip_of_special_tokens(corpus, special_tokens)
    # Pretokenizes based on the regex
    corpus = pretokenize(corpus, pretok_PAT)
    # Splits words into tuples of bytes and counts words
    corpus = split_to_bytes(corpus)

    # Start merges
    for i in tqdm(range(num_of_merges)):
        # If ran the first time, count from all corpus
        if i == 0:
            counts, counts_to_words = count_bytepairs(corpus)
        # If consecutive runs, count only from the changed (merged) words
        else:
            counts, counts_to_words = count_bytepairs(
                corpus, counts, counts_to_words, merged_words, removed_word_freqs
            )
        
        # Check if there are any pairs left to merge
        if not counts:
            # No more pairs to merge, stop early
            break
            
        # Get the most frequent pair
        mf_pair, mf_pair_words = get_mf_pair(counts, counts_to_words)
        # Add merge to merges
        pair_b = pair_to_bytes(mf_pair) # convert to bytes
        merges.append(pair_b)
        # Add the merge to the vocab
        merge_b = "".join(mf_pair)
        vocab[256 + i] = merge_b
        # Apply the merge to the corpus
        corpus, merged_words, removed_word_freqs = merge(corpus, "".join(mf_pair), mf_pair_words)

    # Add special tokens to the vocab
    for i, token in enumerate(special_tokens):
        vocab[256 + num_of_merges + i] = token

    return vocab, merges


def save_vocab_n_merges(vocab, merges):
    # Save everything except the first 255 bytes
    vocab = dict(islice(vocab.items(), 256, max(vocab)))
    # Save vocab as json file
    with open("tinystories_vocab.json", "w") as outfile:
        json.dump(vocab, outfile)
    
    # Prepare merges in expected format
    merges_new = []
    for merge_pair in merges:
        pair = ""
        # Decode and add a space between two merged bytes
        for b in merge_pair:
            pair += b.decode("utf-8") + " "
        merges_new.append(pair + "\n")

    # Write merges to a file
    with open("tinystories_merges.txt", "w") as outfile:
        outfile.writelines(merges_new)


if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 50257, 
                              special_tokens = ["<|endoftext|>", "<start>", "<end>"])

    save_vocab_n_merges(vocab, merges)