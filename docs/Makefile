SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

apidoc: clean-apidoc
	sphinx-apidoc --private --module-first --force \
      --output-dir "$(SOURCEDIR)/api" \
      ../src/revertex ../src/revertex/_version.py

clean-apidoc:
	rm -rf "$(SOURCEDIR)/api"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
