.PHONY: install test fmt fmt-check lint benchmark-rms-norm benchmark-matmul

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=. python3 tests/test_rms_norm.py
	PYTHONPATH=. python3 tests/test_matmul.py
	PYTHONPATH=. python3 tests/test_matmul_autotune.py

fmt:
	black .
	ruff check --fix .

fmt-check:
	black --check .
	ruff check .

benchmark-rms-norm:
	PYTHONPATH=. python3 benchmarks/normalization/benchmark_rms_norm.py --save-plots --plot-dir=docs/plots/normalization/

benchmark-matmul:
	PYTHONPATH=. python3 benchmarks/matmul/benchmark_matmul.py --save-plots --plot-dir=docs/plots/matmul/
