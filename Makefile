.PHONY: install validate test-core

install:
	python -m pip install -e ".[test]"

validate:
	python -m imputeguide validate-config --root .

test-core:
	python -m pytest tests/test_imputeguide_core.py -q
