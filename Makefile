# Makefile with some convenient quick ways to do common things

PROJECT = gammapy
CYTHON ?= cython
version = dev
release = $(version)

help:
	@echo ''
	@echo ' make targets:'
	@echo ''
	@echo '     help               Print this help message (the default)'
	@echo ''
	@echo '     clean              Remove generated files'
	@echo '     clean-repo         Remove all untracked files and directories (use with care!)'
	@echo ''
	@echo '     test               Run pytest'
	@echo '     test-cov           Run pytest with coverage'
	@echo ''
	@echo '     docs-sphinx        Build docs (Sphinx only)'
	@echo '     docs-show          Open local HTML docs in browser'
	@echo ''
	@echo '     trailing-spaces    Remove trailing spaces at the end of lines in *.py files'
	@echo '     black              Run black code formatter'
	@echo '     isort              Run isort code formatter to sort imports'
	@echo '     polish             Run trailing-spaces, black and isort'
	@echo ''
	@echo '     flake8             Run flake8 static code analysis'
	@echo '     pylint             Run pylint static code analysis'
	@echo '     pydocstyle         Run docstring checks'
	@echo ''
	@echo ' Note that most things are done via `python setup.py`, we only use'
	@echo ' make for things that are not trivial to execute via `setup.py`.'
	@echo ''
	@echo ' setup.py commands:'
	@echo ''
	@echo '     python setup.py --help-commands'
	@echo '     python setup.py install'
	@echo '     python setup.py bdist_conda'
	@echo '     python setup.py develop'
	@echo ''
	@echo ' To get info about your Gammapy installation and setup run this command'
	@echo ''
	@echo '     gammapy info'
	@echo ''
	@echo ' For this to work, Gammapy needs to be installed and on your PATH.'
	@echo ' If it is not, then use this equivalent command:'
	@echo ''
	@echo '     python -m gammapy info'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * Gammapy code: https://github.com/gammapy/gammapy'
	@echo ' * Gammapy docs: https://docs.gammapy.org/'
	@echo ''
	@echo ' Most common commands to hack on Gammapy:'
	@echo ''
	@echo '     make help          Print help message with all commands'
	@echo '     pip install -e .   Install Gammapy in editable mode'
	@echo '     gammapy info       Check install and versions'
	@echo '     make clean         Remove auto-generated files'
	@echo '     pytest             Run Gammapy tests (give folder or filename and options)'
	@echo '     make test-cov      Run all tests and measure coverage'
	@echo '     make docs-sphinx   Build documentation locally'
	@echo ''

clean:
	rm -rf build dist docs/_build docs/api temp/ docs/_static/notebooks \
	  htmlcov MANIFEST v gammapy.egg-info .eggs .coverage .cache .pytest_cache \
	  docs/modeling/gallery docs/tutorials
	find . -name ".ipynb_checkpoints" -prune -exec rm -rf {} \;
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.reg" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find gammapy -name '*.c' -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

clean-repo:
	@git clean -f -x -d

test:
	python -m pytest -v gammapy

test-cov:
	python -m pytest -v gammapy --cov=gammapy --cov-report=html

docs-sphinx:
	cd docs && python -m sphinx . _build/html -b html -j auto

docs-show:
	python docs/serve.py

trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

black:
	black $(PROJECT)

isort:
	isort -rc gammapy examples docs -s docs/conf.py

polish: black isort trailing-spaces;

# Note: flake8 is very fast and almost never has false positives
flake8:
	flake8 $(PROJECT)

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
# Note: pylint is very thorough, but slow, and has false positives or nitpicky stuff
pylint:
	pylint -E $(PROJECT)/ \
	--ignore=gammapy/extern \
	-d E0611,E1101,E1103 \
	--msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'

# TODO: fix and re-activate check for the following:
# D103: Missing docstring in public function
# D202: No blank lines allowed after function docstring (found 1)
# D205: 1 blank line required between summary line and description (found 0)
# D400: First line should end with a period (not ')')
# D401: First line should be in imperative mood; try rephrasing (found 'Function')
# D403: First word of the first line should be properly capitalized ('Add', not 'add')
pydocstyle:
	pydocstyle $(PROJECT) \
	--convention=numpy \
	--match-dir='^(?!extern).*' \
	--match='(?!test_).*\.py' \
	--add-ignore=D100,D102,D103,D104,D105,D200,D202,D205,D400,D401,D403,D410

# Note: codespell will pick its options from setup.cfg
codespell:
	codespell gammapy

# TODO: add test and code quality checks for `examples`
