# Makefile with some convenient quick ways to do common things

PROJECT = gammapy
CYTHON ?= cython

help:
	@echo ''
	@echo ' Gammapy available make targets:'
	@echo ''
	@echo '     help             Print this help message (the default)'
	@echo ''
	@echo '     doc-show         Open local HTML docs in browser'
	@echo '     clean            Remove generated files'
	@echo '     clean-repo       Remove all untracked files and directories (use with care!)'
	@echo '     cython           Compile cython files'
	@echo ''
	@echo '     trailing-spaces  Remove trailing spaces at the end of lines in *.py files'
	@echo '     code-analysis    Run static code analysis (flake8 and pylint)'
	@echo '     flake8           Run static code analysis (flake8)'
	@echo '     pylint           Run static code analysis (pylint)'
	@echo '     pydocstyle       Run docstring checks'
	@echo ''
	@echo ' Note that most things are done via `python setup.py`, we only use'
	@echo ' make for things that are not trivial to execute via `setup.py`.'
	@echo ''
	@echo ' Common `setup.py` commands:'
	@echo ''
	@echo '     python setup.py --help-commands'
	@echo '     python setup.py install'
	@echo '     python setup.py develop'
	@echo '     python setup.py test -V'
	@echo '     python setup.py test --help # to see available options'
	@echo '     python setup.py build_docs # use `-l` for clean build'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * Gammapy code: https://github.com/gammapy/gammapy'
	@echo ' * Gammapy docs: http://docs.gammapy.org/'
	@echo ''
	@echo ' Environment:'
	@echo ''
	@echo '     GAMMAPY_EXTRA = $(GAMMAPY_EXTRA)'
	@echo ''
	@echo ' To get more info about your Gammapy installation and setup run this command'
	@echo ''
	@echo '     gammapy info'
	@echo ''
	@echo ' For this to work, Gammapy needs to be installed and on your PATH.'
	@echo ' If it is not, then use this equivalent command:'
	@echo ''
	@echo '     python -m gammapy info'

clean:
	rm -rf build dist docs/_build docs/api docs/notebooks docs/_static/notebooks htmlcov MANIFEST v gammapy.egg-info .eggs .coverage .cache
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find gammapy -name '*.c' -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

clean-repo:
	@git clean -f -x -d

cython:
	find $(PROJECT) -name "*.pyx" -exec $(CYTHON) {} \;

trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

code-analysis: flake8 pylint

# Note: flake8 is very fast and almost never has false positives
flake8:
	flake8 $(PROJECT) \
    --exclude=gammapy/extern,gammapy/conftest.py,gammapy/_astropy_init.py,__init__.py \
    --ignore=E501

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
# Note: pylint is very thorough, but slow, and has false positives or nitpicky stuff
pylint:
	pylint -E $(PROJECT)/ \
	--ignore=_astropy_init.py -f colorized \
	-d E1103,E0611,E1101 \
	--msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'

# TODO: fix or silence all the issue (and check which codes we really want to ignore if any)
# TODO: figure out how exclude works (see https://github.com/PyCQA/pydocstyle/issues/175):
# Should exclude: gammapy/version.py, gammapy/extern
#	       --match-dir='[^gammapy/extern]' \

pydocstyle:
	pydocstyle $(PROJECT) \
	--convention=numpy \
	--add-ignore=D410,D104,D102,D100,D200

# TODO: add test and code quality checks for `examples`

doc-show:
	open docs/_build/html/index.html

test-notebooks:
	which python
	python setup.py install --user
	python test_notebooks.py

conda:
	python setup.py bdist_conda
