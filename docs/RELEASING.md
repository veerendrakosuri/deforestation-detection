# Releasing

The public repository is named `deforestation-detection`. The unique PyPI
distribution is `forest-change-toolkit`, the Python import is
`deforestation_detection`, and the preferred console command is `forest-change`.

## Pre-release checks

Run from a clean checkout:

```powershell
pip install -e ".[dev]"
python -m pytest
ruff check .
python -m build
python -m twine check dist/*
```

Install the wheel into a fresh environment and run these commands from outside the
repository:

```powershell
forest-change --version
forest-change validate-config
forest-change demo --output demo-output
```

Confirm that the version agrees in `pyproject.toml`,
`src/deforestation_detection/__init__.py`, `CITATION.cff`, and `CHANGELOG.md`.

## GitHub release

Create an annotated `vX.Y.Z` tag only after CI passes. The GitHub release should include
the matching wheel and source archive from `dist/`, plus the relevant changelog section.
Pushing the tag also builds a versioned CPU image in GitHub Container Registry.
Verify both the container workflow and release assets before announcing the release.

## PyPI trusted publishing

The repository includes a manual `publish.yml` workflow using PyPI's trusted-publisher
authentication. Before its first run, the owner must create the PyPI project or pending
publisher for `forest-change-toolkit` and authorize:

- GitHub owner: `veerendrakosuri`
- Repository: `deforestation-detection`
- Workflow: `publish.yml`
- Environment: `pypi`

After that one-time account-side setup, run **Publish package** from GitHub Actions.
Do not store a PyPI password or long-lived API token in the repository.
