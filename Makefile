PYTHON=3.8
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	pytest src --flake8 --pylint --mypy
