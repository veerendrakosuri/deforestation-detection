# Contributing

Thanks for helping improve the project. Contributions are welcome in the form of
bug reports, documentation, tests, reproducible datasets, and code changes.

## Before opening a change

1. Search existing issues and explain the user problem before proposing a large feature.
2. Never commit satellite exports, credentials, personal data, or model checkpoints.
3. Keep scientific claims proportional to the evidence and disclose the evaluation region.
4. Add or update tests for behavior changes.

## Local development

```powershell
git clone https://github.com/veerendrakosuri/deforestation-detection.git
cd deforestation-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[geo,ml,dev]"
python -m pytest
ruff check .
```

Smaller documentation or core-array changes may use `pip install -e ".[dev]"`.
Optional dependency tests skip automatically when their feature group is absent.

## Pull requests

- Keep each pull request focused on one problem.
- Describe the test data and commands used for verification.
- Document new CLI options in `docs/CLI_REFERENCE.md`.
- Update `CHANGELOG.md` for user-visible changes.
- Do not report a metric without identifying the split, geography, and reference labels.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
