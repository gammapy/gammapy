# Makefile with some convenient quick ways to do common things

help:
	@echo ''
	@echo 'Gammapy available make targets:'
	@echo ''
	@echo '  help             Print this help message (the default)'
	@echo '  clean            Remove generated files'
	@echo '  clean-repo       Remove all untracked files and directories (use with care!)'
	@echo '  trailing-spaces  Remove trailing spaces at the end of lines in *.py files'
	@echo '  cython           Compile cython files'
	@echo ''
	@echo 'Note that many things are done via `python setup.py`, see'
	@echo '  $ python setup.py --help-commands'
	@echo ''
	@echo 'Probably the most common are to run the tests:'
	@echo '  python setup.py test -V'
	@echo '  python setup.py test --help # to see available options'
	@echo ''
	@echo 'And to build the docs'
	@echo '  python setup.py build_sphinx # use `-l` for clean build'
	@echo '  open docs/_build/html/index.html'
	@echo ''

clean:
	rm -rf build docs/_build docs/api htmlcov

clean-repo:
	@git clean -f -x -d

trailing-spaces:
	find -name "*.py" -exec perl -p -i -e 's/[ \t]*$$//' {} \;

cython:
	find -name "*.pyx" -exec $(CYTHON) {} \;
