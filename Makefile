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

clean:
	rm -rf build docs/_build docs/api htmlcov
	# TODO: do we need to clean CYTHON stuff???!!!

clean-repo:
	@git clean -f -x -d

trailing-spaces:
	find -name "*.py" -exec perl -p -i -e 's/[ \t]*$$//' {} \;

cython:
	find -name "*.pyx" -exec $(CYTHON) {} \;
