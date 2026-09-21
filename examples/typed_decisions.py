#!/usr/bin/env python3
"""State + typed questions -> multiple structured decisions from ONE forward pass.

    uv run python examples/typed_decisions.py
"""

from __future__ import annotations

from laya_play import answer_lines, banner, describe_runtime, load_agent, predict_timed
from laya_play.questions import REFUND_TICKET, TYPED_QUESTIONS


def main() -> int:
    banner("Typed Decisions: choice + score + noul")

    agent = load_agent()
    print("Runtime:")
    print(describe_runtime(agent))

    print("\nQuestions sent in one request:")
    for name, definition in TYPED_QUESTIONS.items():
        print(f"  {name:<24}{definition['type']}")

    print("\nState:")
    print(REFUND_TICKET["body"])

    result, ms = predict_timed(agent, REFUND_TICKET, TYPED_QUESTIONS)
    answers = result["answers"]

    print("\nDecisions (single forward pass):")
    for name in TYPED_QUESTIONS:
        print()
        for line in answer_lines(name, answers[name]):
            print(line)

    print("\nLatency:")
    print(f"{ms:.2f} ms for {len(TYPED_QUESTIONS)} questions "
          f"({ms / len(TYPED_QUESTIONS):.2f} ms/question)")
    print("\nTokens used:", result["usage"])
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
