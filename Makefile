# Makefile with some convenient quick ways to do common things

PROJECT = gammapy
CYTHON ?= cython

help:
	@echo ''
	@echo 'Gammapy available make targets:'
	@echo ''
	@echo '  help             Print this help message (the default)'
	@echo ''
	@echo '  clean            Remove generated files'
	@echo '  clean-repo       Remove all untracked files and directories (use with care!)'
	@echo '  cython           Compile cython files'
	@echo ''
	@echo '  trailing-spaces  Remove trailing spaces at the end of lines in *.py files'
	@echo '  code-analysis    Run code analysis (flake8 and pylint)'
	@echo '  flake8           Run code analysis (flake8)'
	@echo '  pylint           Run code analysis (pylint)'
	@echo ''
	@echo 'Note that most things are done via `python setup.py`, we only use'
	@echo 'make for things that are not trivial to execute via `setup.py`.'
	@echo ''
	@echo 'Here'\''s some very common `setup.py` commands:'
	@echo ''
	@echo '  python setup.py --help-commands'
	@echo '  python setup.py install'
	@echo '  python setup.py develop'
	@echo '  python setup.py test -V'
	@echo '  python setup.py test --help # to see available options'
	@echo '  python setup.py build_sphinx # use `-l` for clean build'
	@echo '  open docs/_build/html/index.html'
	@echo ''

clean:
	rm -rf build docs/_build docs/api htmlcov
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

clean-repo:
	@git clean -f -x -d

cython:
	find $(PROJECT) -name "*.pyx" -exec $(CYTHON) {} \;

trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

code-analysis: flake8 pylint

flake8:
	flake8 $(PROJECT) | grep -v __init__ | grep -v external

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
pylint:
	pylint -E $(PROJECT)/ -d E1103,E0611,E1101 \
	       --ignore=_astropy_init.py -f colorized \
	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'

# TODO: add test and code quality checks for `examples`

gammapy-extra: 
	git clone git@github.com:gammapy/gammapy-extra.git

# TODO: add test for IPython notebooks in gammapy-extra
# TODO: probably best to put this in a Python script?
test-notebooks: gammapy-extra
	# For now just run one example ... should run all
	runipy gammapy-extra/notebooks/Index.ipynb 
	runipy gammapy-extra/notebooks/source_catalogs.ipynb
