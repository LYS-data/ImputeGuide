.PHONY: install validate test

install:
	python -m pip install -e ".[test]"

validate:
	python -m imputeguide validate-config --root .

test:
	python -m pytest -q
