# Laya Multilingual Playground — WSL + CPU.
# USE_TF=0 keeps transformers from probing TensorFlow, which can deadlock model loading.
USE_TF := 0
export USE_TF

UV ?= uv

.PHONY: help setup download demo typed compare benchmark test clean

help:
	@echo "make setup      - create .venv (Python 3.11) and install dependencies"
	@echo "make download   - pull convaiinnovations/laya-multilingual from ModelScope into models/"
	@echo "make demo       - Chinese ticket -> choice decision  (examples/basic_multilingual.py)"
	@echo "make typed      - one forward pass -> choice + score + noul"
	@echo "make compare    - 10 Chinese scenarios side by side"
	@echo "make benchmark  - local WSL cold-start + warm-inference latency"
	@echo "make test       - pytest smoke tests"

setup:
	$(UV) venv --python 3.11
	$(UV) sync --extra dev

download:
	$(UV) run python scripts/download_model.py

demo:
	$(UV) run python examples/basic_multilingual.py

typed:
	$(UV) run python examples/typed_decisions.py

compare:
	$(UV) run python examples/compare_inputs.py

benchmark:
	$(UV) run python examples/benchmark.py

test:
	$(UV) run pytest -v

clean:
	rm -rf .pytest_cache **/__pycache__
