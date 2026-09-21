"""Smoke test: ModelScope checkpoint on disk -> laya loads it -> Chinese decision comes back.

Run with `make test` (needs `make download` first). Structure is asserted, never values:
the checkpoint ships uncalibrated, so probabilities are not stable ground truth.
"""

from __future__ import annotations

import pytest

from laya_play import (
    MODEL_ID,
    REQUIRED_FILES,
    describe_runtime,
    load_agent,
    missing_files,
    model_dir,
)
from laya_play.questions import REFUND_TICKET, TYPED_QUESTIONS


def test_laya_imports():
    import laya

    assert laya.__version__
    assert callable(laya.load)


def test_model_dir_exists():
    assert model_dir().is_dir(), f"{model_dir()} missing - run `make download`"


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_checkpoint_file_exists(relative):
    assert (model_dir() / relative).is_file(), f"{relative} missing - run `make download`"


def test_no_missing_files_reported():
    assert missing_files() == []


@pytest.fixture(scope="session")
def agent():
    if missing_files():
        pytest.fail(f"checkpoint incomplete ({missing_files()}) - run `make download` first")
    return load_agent()


def test_agent_reports_its_device(agent):
    assert describe_runtime(agent).split()[0] in {"CPU", "CUDA", "MPS"}


def test_predict_returns_structured_answers(agent):
    result = agent.predict(REFUND_TICKET, TYPED_QUESTIONS)

    assert result is not None
    assert isinstance(result, dict)
    assert "answers" in result and isinstance(result["answers"], dict)
    assert set(result["answers"]) == set(TYPED_QUESTIONS)

    department = result["answers"]["department"]
    assert department["type"] == "choice"
    assert department["choice"] in TYPED_QUESTIONS["department"]["criteria"]
    assert set(department["probabilities"]) == set(TYPED_QUESTIONS["department"]["criteria"])
    assert 0.0 <= department["confidence"] <= 1.0

    urgency = result["answers"]["urgency"]
    assert urgency["type"] == "score"
    assert 0.0 <= urgency["score"] <= len(TYPED_QUESTIONS["urgency"]["criteria"]) - 1

    ai = result["answers"]["ai_deployment_related"]
    assert ai["type"] == "noul"
    assert 0.0 <= ai["noul"] <= 1.0


def test_chinese_input_produces_a_known_label(agent):
    """The one behavioural check: a refund ticket is a decision, not an error or a hallucination."""
    result = agent.predict(REFUND_TICKET, {"department": TYPED_QUESTIONS["department"]})
    answer = result["answers"]["department"]
    assert answer["choice"] in {"billing", "other"}, answer
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3


def test_model_id_matches_the_modelscope_repo():
    assert MODEL_ID == "convaiinnovations/laya-multilingual"
