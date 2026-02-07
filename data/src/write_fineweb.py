from datasets import load_dataset
import os
from multiprocessing import Process, cpu_count


import typing


def process_shard(
    world_size: int, rank: int, ds: typing.Any, out_path: str | os.PathLike
) -> None:

    shard = ds.shard(world_size, rank)

    print(f"Process {rank} started...")
    with open(out_path, 'w', encoding="utf-8") as f_out:
        for t in shard['text']:
            f_out.write(t + "\n")


def merge_shards(
    shard_files: list[str | os.PathLike], final_output: str | os.PathLike
) -> None:
    print("Merging shards...")
    with open(final_output, "w", encoding="utf-8") as f_out:
        for s_file in shard_files:
            with open(s_file, 'r', encoding="utf-8") as f_in:
                while True:
                    chunk = f_in.read(1024 * 1024 * 100)
                    if not chunk:
                        break
                    f_out.write(chunk)
            os.remove(s_file)

    print(f"Merged into: {final_output}")


if __name__ == "__main__":
    fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    num_workers = cpu_count() // 2
    total_shards = 15
    out_path = "data/fineweb.txt"

    shard_files = []
    processes = []
    for i in range(num_workers):
        output_shard = f"temp_shard{i}.txt"
        shard_files.append(output_shard)

        p = Process(target=process_shard, args=(total_shards, i, fw, output_shard))
        p.start()
        processes.append(p)

    print("Writing to the file now...")
    for p in processes:
        p.join()
    
    merge_shards(shard_files, out_path)