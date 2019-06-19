dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test: dev check docs
	pip install -e .
	pytest tests

check:
	isort -rc -c src/
	isort -rc -c tests/
	black src/ --check
	black tests/ --check
	flake8 src/
	flake8 tests/

docs:
	sphinx-build -W -b html docs/ _build/html

clean:
	pip uninstall -y emmental
	rm -rf src/emmental.egg-info
	rm -rf _build/

.PHONY: dev test clean check docs
