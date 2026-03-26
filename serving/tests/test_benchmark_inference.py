import json

import pytest

from serving import benchmark_inference


def test_load_prompt_cases_from_file(tmp_path):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            [
                {
                    "name": "example",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Say hello."},
                    ],
                }
            ]
        )
    )

    cases = benchmark_inference.load_prompt_cases(str(prompt_file))

    assert cases == [
        {
            "name": "example",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hello."},
            ],
        }
    ]


def test_summarize_runs_computes_median_and_p95():
    runs = [
        {
            "run_index": 0,
            "prompt_tokens": 8,
            "generated_tokens": 10,
            "latency_seconds": 1.0,
            "decode_tokens_per_second": 10.0,
            "total_tokens_per_second": 18.0,
            "peak_gpu_memory_bytes": None,
        },
        {
            "run_index": 1,
            "prompt_tokens": 8,
            "generated_tokens": 12,
            "latency_seconds": 2.0,
            "decode_tokens_per_second": 6.0,
            "total_tokens_per_second": 10.0,
            "peak_gpu_memory_bytes": None,
        },
        {
            "run_index": 2,
            "prompt_tokens": 8,
            "generated_tokens": 14,
            "latency_seconds": 4.0,
            "decode_tokens_per_second": 3.5,
            "total_tokens_per_second": 5.5,
            "peak_gpu_memory_bytes": None,
        },
    ]

    summary = benchmark_inference.summarize_runs(runs)

    assert summary["run_count"] == 3
    assert summary["prompt_tokens"] == 8
    assert summary["median_generated_tokens"] == 12
    assert summary["median_latency_seconds"] == 2.0
    assert summary["p95_latency_seconds"] == pytest.approx(3.8)
    assert summary["median_decode_tokens_per_second"] == 6.0
    assert summary["p95_decode_tokens_per_second"] == pytest.approx(9.6)
    assert summary["median_total_tokens_per_second"] == 10.0
    assert summary["median_peak_gpu_memory_bytes"] is None


def test_build_result_payload_includes_startup_and_case_summaries():
    payload = benchmark_inference.build_result_payload(
        repo_id="repo/example",
        device="cpu",
        warmup_runs=2,
        runs=5,
        max_new_tokens=32,
        temp=0.7,
        top_p=0.8,
        startup_seconds=1.25,
        overall={
            "median_latency_seconds": 0.5,
            "p95_latency_seconds": 0.7,
            "median_decode_tokens_per_second": 20.0,
            "median_total_tokens_per_second": 24.0,
        },
        cases=[
            {
                "name": "short_prompt",
                "summary": {
                    "median_latency_seconds": 0.4,
                    "p95_latency_seconds": 0.5,
                    "median_decode_tokens_per_second": 22.0,
                    "median_total_tokens_per_second": 28.0,
                },
                "runs": [],
            }
        ],
    )

    assert payload["metadata"]["repo_id"] == "repo/example"
    assert payload["metadata"]["device"] == "cpu"
    assert payload["metadata"]["startup_mode"] == "cached"
    assert payload["startup"] == {"startup_seconds": 1.25}
    assert payload["overall"]["median_latency_seconds"] == 0.5
    assert payload["cases"][0]["name"] == "short_prompt"
    assert payload["cases"][0]["summary"]["median_decode_tokens_per_second"] == 22.0
