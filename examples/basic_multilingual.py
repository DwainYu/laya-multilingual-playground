#!/usr/bin/env python3
"""One Chinese support ticket -> one `choice` decision with probabilities.

    uv run python examples/basic_multilingual.py
"""

from __future__ import annotations

import json

from laya_play import describe_runtime, load_agent, predict_timed, banner
from laya_play.questions import REFUND_TICKET, TRIAGE_QUESTIONS


def main() -> int:
    banner("Laya Multilingual Demo")

    agent = load_agent()
    print("Runtime:")
    print(describe_runtime(agent))

    print("\nInput:")
    print(REFUND_TICKET["body"])

    result, ms = predict_timed(agent, REFUND_TICKET, TRIAGE_QUESTIONS)
    answer = result["answers"]["department"]

    print("\nDecision:")
    print(f"department = {answer['choice']}")

    print("\nConfidence:")
    print(answer["confidence"])

    print("\nProbabilities:")
    width = max(len(k) for k in answer["probabilities"])
    for label, prob in answer["probabilities"].items():
        print(f"{label:<{width}} {prob:.4f}")

    print("\nLatency:")
    print(f"{ms:.2f} ms")

    print("\nStatus:")
    print("PASS")
    print("=" * 40)

    print("\n--- full result (structure returned by agent.predict) ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
