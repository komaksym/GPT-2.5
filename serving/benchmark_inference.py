import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from serving.inference import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMP,
    DEFAULT_TOP_P,
    generate_response,
    get_model_repo_id,
    load_inference_resources,
)

BUILTIN_CASES = [
    {
        "name": "short_prompt",
        "messages": [
            {"role": "user", "content": "Explain GPT-2 in one sentence."},
        ],
    },
    {
        "name": "coding_prompt",
        "messages": [
            {
                "role": "user",
                "content": "Write a short Python function that checks whether a string is a palindrome.",
            },
        ],
    },
    {
        "name": "longer_prompt",
        "messages": [
            {
                "role": "system",
                "content": "You are a concise teaching assistant.",
            },
            {
                "role": "user",
                "content": (
                    "Summarize the tradeoffs between autoregressive language models and "
                    "masked language models in a short paragraph."
                ),
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cached startup and steady-state inference for the serving stack. "
            "Startup timing measures cached model/tokenizer load only and excludes first-time downloads."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=get_model_repo_id(),
        help="Model repository id to load for startup and generation benchmarking.",
    )
    parser.add_argument(
        "--prompt-file",
        help=(
            "Optional JSON file containing benchmark prompt cases. "
            "If omitted, the built-in prompt set is used."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of untimed warmup generations to run before measured inference runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measured inference runs to record for each prompt case.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate per benchmark run.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP,
        help="Sampling temperature used during benchmarked generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p nucleus sampling threshold used during benchmarked generation.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the benchmark results as formatted JSON.",
    )
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be 0 or greater.")
    if args.runs < 1:
        parser.error("--runs must be at least 1.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")
    if args.temp <= 0:
        parser.error("--temp must be greater than 0.")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be between 0 and 1.")
    return args


def load_prompt_cases(prompt_file: str | None) -> list[dict[str, object]]:
    if prompt_file is None:
        return BUILTIN_CASES

    data = json.loads(Path(prompt_file).read_text())
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON array of prompt cases.")
    if not data:
        raise ValueError("Prompt file must contain at least one prompt case.")

    cases = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt case {index} must be an object.")

        name = item.get("name")
        messages = item.get("messages")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Prompt case {index} must have a non-empty string name.")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Prompt case {index} must have a non-empty messages list."
            )

        normalized_messages = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Prompt case {index} message {message_index} must be an object."
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    f"Prompt case {index} message {message_index} must have string role and content."
                )
            normalized_messages.append({"role": role, "content": content})

        cases.append({"name": name, "messages": normalized_messages})

    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    latencies = [float(run["latency_seconds"]) for run in runs]
    decode_rates = [float(run["decode_tokens_per_second"]) for run in runs]
    total_rates = [float(run["total_tokens_per_second"]) for run in runs]
    generated_tokens = [int(run["generated_tokens"]) for run in runs]
    peak_memories = [
        int(run["peak_gpu_memory_bytes"])
        for run in runs
        if run["peak_gpu_memory_bytes"] is not None
    ]

    return {
        "run_count": len(runs),
        "prompt_tokens": int(runs[0]["prompt_tokens"]),
        "median_generated_tokens": median(generated_tokens),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "median_decode_tokens_per_second": median(decode_rates),
        "p95_decode_tokens_per_second": percentile(decode_rates, 0.95),
        "median_total_tokens_per_second": median(total_rates),
        "median_peak_gpu_memory_bytes": (
            int(median(peak_memories)) if peak_memories else None
        ),
    }


def build_result_payload(
    *,
    repo_id: str,
    device: str,
    warmup_runs: int,
    runs: int,
    max_new_tokens: int,
    temp: float,
    top_p: float,
    startup_seconds: float,
    cases: list[dict[str, object]],
    overall: dict[str, object],
) -> dict[str, object]:
    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "repo_id": repo_id,
            "device": device,
            "warmup_runs": warmup_runs,
            "runs": runs,
            "max_new_tokens": max_new_tokens,
            "temp": temp,
            "top_p": top_p,
            "startup_mode": "cached",
        },
        "startup": {"startup_seconds": startup_seconds},
        "overall": overall,
        "cases": cases,
    }


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prompt_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return int(input_ids.shape[-1])


def generated_token_count(tokenizer, response: str) -> int:
    return len(tokenizer.encode(response, add_special_tokens=False))


def run_case(
    *,
    case: dict[str, object],
    resources,
    warmup_runs: int,
    runs: int,
    max_new_tokens: int,
    temp: float,
    top_p: float,
) -> dict[str, object]:
    messages = case["messages"]
    prompt_tokens = prompt_token_count(resources.tokenizer, messages)

    for _ in range(warmup_runs):
        generate_response(
            messages=messages,
            resources=resources,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
        )

    measured_runs = []
    for index in range(runs):
        if resources.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resources.device)
        maybe_synchronize(resources.device)

        started_at = perf_counter()
        response = generate_response(
            messages=messages,
            resources=resources,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
        )
        maybe_synchronize(resources.device)
        latency_seconds = perf_counter() - started_at

        generated_tokens = generated_token_count(resources.tokenizer, response)
        peak_memory = None
        if resources.device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(resources.device))

        measured_runs.append(
            {
                "run_index": index,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "latency_seconds": latency_seconds,
                "decode_tokens_per_second": (
                    generated_tokens / latency_seconds if latency_seconds else 0.0
                ),
                "total_tokens_per_second": (
                    (prompt_tokens + generated_tokens) / latency_seconds
                    if latency_seconds
                    else 0.0
                ),
                "peak_gpu_memory_bytes": peak_memory,
            }
        )

    return {
        "name": case["name"],
        "messages": messages,
        "summary": summarize_runs(measured_runs),
        "runs": measured_runs,
    }


def print_summary(startup_seconds: float, cases: list[dict[str, object]]) -> None:
    print(f"Startup: {startup_seconds:.3f}s (cached load)")

    all_runs = []
    for case in cases:
        summary = case["summary"]
        all_runs.extend(case["runs"])

        line = (
            f"{case['name']}: median {summary['median_latency_seconds']:.3f}s"
            f" | p95 {summary['p95_latency_seconds']:.3f}s"
            f" | decode {summary['median_decode_tokens_per_second']:.2f} tok/s"
            f" | total {summary['median_total_tokens_per_second']:.2f} tok/s"
            f" | prompt {summary['prompt_tokens']} tok"
        )
        if summary["median_peak_gpu_memory_bytes"] is not None:
            peak_mib = summary["median_peak_gpu_memory_bytes"] / (1024**2)
            line += f" | peak {peak_mib:.1f} MiB"
        print(line)

    overall = summarize_runs(all_runs)
    print(
        "Overall:"
        f" median {overall['median_latency_seconds']:.3f}s"
        f" | p95 {overall['p95_latency_seconds']:.3f}s"
        f" | decode {overall['median_decode_tokens_per_second']:.2f} tok/s"
        f" | total {overall['median_total_tokens_per_second']:.2f} tok/s"
    )


def main() -> None:
    args = parse_args()
    cases = load_prompt_cases(args.prompt_file)

    started_at = perf_counter()
    resources = load_inference_resources(args.repo_id)
    maybe_synchronize(resources.device)
    startup_seconds = perf_counter() - started_at

    case_results = [
        run_case(
            case=case,
            resources=resources,
            warmup_runs=args.warmup_runs,
            runs=args.runs,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            top_p=args.top_p,
        )
        for case in cases
    ]

    print_summary(startup_seconds, case_results)

    overall = summarize_runs([run for case in case_results for run in case["runs"]])
    if args.output:
        payload = build_result_payload(
            repo_id=args.repo_id,
            device=str(resources.device),
            warmup_runs=args.warmup_runs,
            runs=args.runs,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            top_p=args.top_p,
            startup_seconds=startup_seconds,
            cases=case_results,
            overall=overall,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
