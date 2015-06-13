MODULE = gammapy

CURDIR = $(shell pwd)

build:
	python $(CURDIR)/setup.py develop

test: build
	python $(CURDIR)/setup.py test

doc:
	python $(CURDIR)/setup.py build_sphinx

all: build test doc

clean:
	rm -rf $(CURDIR)/build $(CURDIR)/docs/_build $(CURDIR)/docs/api $(CURDIR)/htmlcov

distclean: clean
	python $(CURDIR)/setup.py develop -u
