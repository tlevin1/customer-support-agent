.PHONY: eval install test

install:
	pip install -r requirements.txt

eval:
	python eval.py

test:
	python -m pytest test_triage.py -v
