PYTHON ?= python3

.PHONY: test docs-check completeness completeness-live completeness-all build publish-test publish

test:
	$(PYTHON) -m unittest discover -s tests -v

docs-check:
	$(PYTHON) examples/validate_examples.py

completeness:
	$(PYTHON) completeness/runner.py --mode fixture --fail-under 1.0

completeness-live:
	$(PYTHON) completeness/runner.py --mode live --fail-under 0.0

completeness-all:
	$(PYTHON) completeness/runner.py --mode all --fail-under 1.0

build:
	uv run python -m build

publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*
