"""
Dataset Tokenization Script

This script tokenizes a text dataset using the GPT-2 tokenizer (via tiktoken)
and saves the result as a binary file of uint16 tokens. It uses multiprocessing
to parallelize the work across multiple CPU cores for faster processing.

The output binary file can be memory-mapped during training for efficient data loading.
"""

import tiktoken
import numpy as np
import os
from multiprocessing import Process, cpu_count

def process_shard(
    input_file: str | os.PathLike,
    start_byte: int,
    end_byte: int,
    worker_id: int,
    output_file: str | os.PathLike,
) -> None:
    """
    Process a portion (shard) of the input file.
    
    Each worker processes a byte range of the input file, tokenizes the text,
    and writes the tokens to a temporary shard file.
    
    Args:
        input_file: Path to the input text file
        start_byte: Starting byte position for this worker
        end_byte: Ending byte position for this worker
        worker_id: Identifier for this worker (for logging)
        output_file: Path to write the tokenized output shard
    """
    # Initialize the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Buffer size for accumulating tokens before writing (50M tokens)
    # Writing in larger chunks is more efficient than writing line by line
    buffer_size = 50000000 # About 100MB of tokens
    token_buffer = []

    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        # Seek to the starting position for this worker
        f_in.seek(start_byte)

        # If not starting from the beginning, skip to the next complete line
        # This avoids processing partial lines that might start mid-word
        if start_byte != 0:
            f_in.readline()

        # Process lines until we reach our assigned end position
        while f_in.tell() < end_byte:
            line = f_in.readline()
            if not line:
                break

            tokens = enc.encode(line.decode("utf-8"), allowed_special={'<|endoftext|>'})
            # Add tokens to the buffer
            token_buffer.extend(tokens)

            # When buffer is full, convert to numpy array and write to disk
            # Using uint16 since GPT-2 vocab size (50257) fits in 16 bits
            if len(token_buffer) >= buffer_size:
                arr = np.array(token_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_buffer = []

        # Flush any remaining tokens in the buffer after processing all lines
        if token_buffer:
            arr = np.array(token_buffer, dtype=np.uint16)
            f_out.write(arr.tobytes())
            token_buffer = []

    print(f"Worker {worker_id} finished...")


def merge_shards(
    shard_files: list[str | os.PathLike], final_output: str | os.PathLike
) -> None:
    """
    Merge all temporary shard files into a single output file.
    
    After all workers finish processing their portions, this function
    concatenates the binary shard files in order and cleans up the temp files.
    
    Args:
        shard_files: List of paths to temporary shard files (in order)
        final_output: Path to the final merged output file
    """
    print("Merging shards...")
    with open(final_output, "wb") as f_out:
        for s_file in shard_files:
            with open(s_file, "rb") as f_in:
                # Read and write in 100MB chunks for memory efficiency
                while True:
                    chunk = f_in.read(1024 * 1024 * 100) # 100 MB copy buffer
                    if not chunk:
                        break
                    f_out.write(chunk)
            # Remove the temporary shard file after merging
            os.remove(s_file)
    print(f"Merged into {final_output}")


def main() -> None:
    """
    Main entry point for parallel dataset tokenization.
    
    Splits the input file into chunks, spawns worker processes,
    and merges the results into a single binary token file.
    """
    # Configuration
    input_path = "data/fineweb_train.txt"
    final_output = "fineweb_train.bin"
    
    # Use half the available CPU cores to avoid system overload
    n_workers = cpu_count() // 2
    file_size = os.path.getsize(input_path)

    # Calculate byte range for each worker
    chunk_size = file_size // n_workers
    processes = []
    shard_files = []

    print(f"Tokenizing {file_size/1e9:.2f}GB with {n_workers} cores...")

    # Spawn worker processes, each handling a portion of the file
    for i in range(n_workers):
        start = i * chunk_size
        # Last worker handles any remaining bytes due to integer division
        end = file_size if i == n_workers - 1 else (i + 1) * chunk_size

        output_shard = f"temp_shard{i}.bin"
        shard_files.append(output_shard)

        # Create and start a new process for this shard
        p = Process(target=process_shard, args=(input_path, start, end, i, output_shard))
        p.start()
        processes.append(p)
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Combine all shards into the final output file
    merge_shards(shard_files, final_output)


if __name__ == "__main__":
    
    main()