import tiktoken
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Process, cpu_count


def process_shard(input_file, start_byte, end_byte, worker_id, output_file):
    enc = tiktoken.get_encoding("gpt2")

    buffer_size = 50000000
    token_buffer = []

    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_in.seek(start_byte)

        if start_byte != 0:
            f_in.readline()

        while f_in.tell() < end_byte:
            line = f_in.readline()
            if not line:
                break

            tokens = enc.encode(line.decode("utf-8"), allowed_special={'<|endoftext|>'})
            token_buffer.extend(tokens)

            if len(token_buffer) >= buffer_size:
                arr = np.array(token_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_buffer = []

        if token_buffer:
            arr = np.array(token_buffer, dtype=np.uint16)
            f_out.write(arr.tobytes())
            token_buffer = []

    # 4. Save this shard
    print(f"Worker {worker_id} finished...")


def merge_shards(shard_files, final_output):
    print("Merging shards...")
    with open(final_output, "wb") as f_out:
        for s_file in shard_files:
            with open(s_file, "rb") as f_in:
                while True:
                    chunk = f_in.read(1024 * 1024 * 100) # 100 MB copy buffer
                    if not chunk:
                        break
                    f_out.write(chunk)
            os.remove(s_file)
    print(f"Merged into {final_output}")


def main():
    input_path = "data/owt_train.txt"
    final_output = "owt_train.bin"
    n_workers = cpu_count() // 3
    file_size = os.path.getsize(input_path)

    chunk_size = file_size // n_workers
    processes = []
    shard_files = []

    print(f"Tokenizing {file_size/1e9:.2f}GB with {n_workers} cores...")

    for i in range(n_workers):
        start = i * chunk_size
        end = file_size if i == n_workers - 1 else (i + 1) * chunk_size

        output_shard = f"temp_shard{i}.bin"
        shard_files.append(output_shard)

        p = Process(target=process_shard, args=(input_path, start, end, i, output_shard))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    merge_shards(shard_files, final_output)

if __name__ == "__main__":
    main()