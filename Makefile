SHELL := /bin/bash
.ONESHELL:

PYTHON ?= python3
ENVFILE ?= .env
PACKAGE_NAME := vyvoj25_framework
DIST_DIR := dist
NS_BUILD_DIR := vyvoj25_fork_build

.PHONY: all deps bundle build check clean upload upload-test release test-release print-version test-install \
        bundle-ns build-ns check-ns upload-ns release-ns test-install-ns clean-ns \
        submodules-init submodules-update submodules-checkout submodules-status overlay-ls

all: release

# Install local tooling needed to build and upload
deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade build twine tomli

# Bundle upstream sources into vyvoj25_framework/livekit
bundle:
	$(PYTHON) bundle_framework.py

# Bundle namespaced variant into $(NS_BUILD_DIR)/vyvoj25_fork (import as vyvoj25_fork.*)
bundle-ns:
	$(PYTHON) bundle_framework.py ns

# Build sdist and wheel into ./dist/
build: clean
	$(PYTHON) -m build

# Build namespaced sdist and wheel into ./$(NS_BUILD_DIR)/dist/
build-ns: clean-ns bundle-ns
	$(PYTHON) -m build $(NS_BUILD_DIR)

# Validate built artifacts' metadata
check:
	twine check $(DIST_DIR)/*

# Validate namespaced built artifacts' metadata
check-ns:
	twine check $(NS_BUILD_DIR)/dist/*

# Clean build artifacts
clean:
	rm -rf $(DIST_DIR) build *.egg-info

# Clean namespaced build artifacts
clean-ns:
	rm -rf $(NS_BUILD_DIR)/dist $(NS_BUILD_DIR)/build $(NS_BUILD_DIR)/*.egg-info

# Upload to PyPI using credentials in .env (TWINE_USERNAME, TWINE_PASSWORD)
upload:
	if [ -f "$(ENVFILE)" ]; then set -a; . "$(ENVFILE)"; set +a; fi
	if [ -z "$${TWINE_USERNAME:-}" ]; then echo "ERROR: Set TWINE_USERNAME in $(ENVFILE) (usually __token__)"; exit 1; fi
	if [ -z "$${TWINE_PASSWORD:-}" ]; then echo "ERROR: Set TWINE_PASSWORD in $(ENVFILE) (pypi-...)"; exit 1; fi
	twine upload --repository $${PYPI_REPOSITORY:-pypi} $(DIST_DIR)/*

# Upload namespaced dist to PyPI using credentials in .env
upload-ns:
	if [ -f "$(ENVFILE)" ]; then set -a; . "$(ENVFILE)"; set +a; fi
	if [ -z "$${TWINE_USERNAME:-}" ]; then echo "ERROR: Set TWINE_USERNAME in $(ENVFILE) (usually __token__)"; exit 1; fi
	if [ -z "$${TWINE_PASSWORD:-}" ]; then echo "ERROR: Set TWINE_PASSWORD in $(ENVFILE) (pypi-...)"; exit 1; fi
	twine upload --repository $${PYPI_REPOSITORY:-pypi} $(NS_BUILD_DIR)/dist/*

# Upload to TestPyPI using the same credentials (token must have TestPyPI scope)
upload-test:
	if [ -f "$(ENVFILE)" ]; then set -a; . "$(ENVFILE)"; set +a; fi
	TWU=$${TWINE_TEST_USERNAME:-$${TWINE_USERNAME:-}}; \
	TWP=$${TWINE_TEST_PASSWORD:-$${TWINE_PASSWORD:-}}; \
	if [ -z "$$TWU" ] || [ -z "$$TWP" ]; then \
	  echo "ERROR: Provide TWINE_TEST_USERNAME/TWINE_TEST_PASSWORD or fallback TWINE_USERNAME/TWINE_PASSWORD in $(ENVFILE)"; \
	  echo "Hint: TestPyPI requires its own token from https://test.pypi.org/manage/account/token/"; \
	  exit 1; \
	fi; \
	TWINE_USERNAME="$$TWU" TWINE_PASSWORD="$$TWP" \
	  twine upload --repository $${TESTPYPI_REPOSITORY:-testpypi} $(DIST_DIR)/*

# Full release to PyPI
release: deps bundle build check upload

# Full release to TestPyPI
test-release: deps bundle build check upload-test

# Full release of namespaced package to PyPI
release-ns: deps build-ns check-ns upload-ns

# Print version from pyproject.toml
print-version:
	$(PYTHON) -c "import sys, importlib; t=importlib.import_module('tomllib' if sys.version_info>=(3,11) else 'tomli'); print(t.load(open('pyproject.toml','rb'))['project']['version'])"

# Quick install test in a temp venv
test-install:
	$(PYTHON) -m venv .venv-test
	source .venv-test/bin/activate
	python -m pip install --upgrade pip
	python -m pip install $(DIST_DIR)/*.whl
	python -c "import livekit, livekit.agents, livekit.plugins.elevenlabs; print('agents at:', livekit.agents.__file__); print('elevenlabs at:', livekit.plugins.elevenlabs.__file__)"
	deactivate

# Quick co-install test (upstream + namespaced fork) in a temp venv
test-install-ns:
	$(PYTHON) -m venv .venv-test-ns
	source .venv-test-ns/bin/activate
	python -m pip install --upgrade pip
	python -m pip install livekit-agents  # upstream
	python -m pip install $(NS_BUILD_DIR)/dist/*.whl  # namespaced fork
	python -c "import livekit.agents, vyvoj25_fork.agents; print('upstream agents at:', livekit.agents.__file__); print('fork agents at:', vyvoj25_fork.agents.__file__)"
	deactivate

# --- Git submodules helpers (track upstream repos cleanly) ---
submodules-init:
	git submodule update --init --recursive

submodules-update:
	git submodule foreach 'git fetch --tags --prune && git pull --ff-only || true'

# Usage: make submodules-checkout TAG=vX.Y.Z  (or a commit SHA)
submodules-checkout:
	@if [ -z "$(TAG)" ]; then echo "Usage: make submodules-checkout TAG=<tag-or-commit>"; exit 1; fi
	git submodule foreach 'git checkout $(TAG)'

submodules-status:
	git submodule status --recursive

# Inspect overlay contents
overlay-ls:
	@if [ -d overlay ]; then find overlay -type f | sort; else echo "overlay/ (empty)"; fi
