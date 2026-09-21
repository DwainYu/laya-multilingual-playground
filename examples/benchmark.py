#!/usr/bin/env python3
"""Local WSL latency benchmark. Nothing here is an official number.

    uv run python examples/benchmark.py

Measures cold start (checkpoint load) and warm inference (repeat forward passes),
plus how cost scales with questions per request.
"""

from __future__ import annotations

import os
import platform
import time

import torch

from laya_play import banner, describe_runtime, load_agent, predict_timed
from laya_play.questions import DEPARTMENT_CRITERIA, REFUND_TICKET, TYPED_QUESTIONS


def choice_question(options: dict, n: int) -> dict:
    """n independent `choice` questions so the batch grows without blowing the token budget."""
    return {f"q{i}": {"type": "choice", "instructions": "这个工单应该交给哪个部门处理？",
                      "criteria": options} for i in range(n)}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> int:
    banner("Local WSL benchmark (CPU)")
    print("Machine:")
    print(f"  {platform.system()} {platform.release()} · {platform.machine()} · "
          f"{os.cpu_count()} logical CPUs · torch threads={torch.get_num_threads()}")
    print(f"  Python {platform.python_version()} · torch {torch.__version__}")
    print()

    start = time.perf_counter()
    agent = load_agent()
    load_seconds = time.perf_counter() - start
    print("Runtime:")
    print(describe_runtime(agent))
    print(f"\ncold start (laya.load of the local ModelScope checkpoint): {load_seconds:.2f} s")

    single = {"department": TYPED_QUESTIONS["department"]}
    timings = []
    print("\nwarm inference, 1 question per request:")
    for i in range(10):
        _, ms = predict_timed(agent, REFUND_TICKET, single)
        timings.append(ms)
        print(f"  prediction {i + 1:<3}{ms:>8.2f} ms")
    print(f"  {'average':<14}{sum(timings) / len(timings):>8.2f} ms")
    print(f"  {'p50':<14}{percentile(timings, 0.5):>8.2f} ms")
    print(f"  {'min':<14}{min(timings):>8.2f} ms   {'max':<5}{max(timings):.2f} ms")

    print("\nwarm inference, questions per request:")
    for n in (1, 3, 10):
        questions = choice_question(DEPARTMENT_CRITERIA, n)
        _, ms = predict_timed(agent, REFUND_TICKET, questions)
        print(f"  {n:>3} questions: {ms:>8.2f} ms  ({ms / n:.2f} ms/question)")

    print("\nSingle forward pass, all 3 typed questions (choice + score + noul):")
    _, ms = predict_timed(agent, REFUND_TICKET, TYPED_QUESTIONS)
    print(f"  {ms:.2f} ms")
    print("\nOfficial Laya numbers are measured on a T4 GPU (32.8 ms / 1 question) and are")
    print("not comparable to the CPU figures above. See README 'Benchmarks'.")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
