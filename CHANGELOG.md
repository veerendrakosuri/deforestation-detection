# Changelog

All notable user-facing changes are documented here.

## [Unreleased]

## [0.2.0] - 2026-10-25

### Added

- Short `forest-change` command while preserving `deforestation-detection`.
- Unique `forest-change-toolkit` distribution name for PyPI publishing.
- `init`, `validate-config`, and `doctor` commands.
- Bundled configuration that works outside a source checkout.
- PyPI-ready metadata, package data, Docker support, and distribution checks.
- CLI reference, contribution guide, security policy, conduct policy, and citation metadata.
- Tests for installed-package configuration and developer onboarding commands.

### Changed

- Expected CLI failures now use concise messages and meaningful exit statuses.
- Heavy geospatial and machine-learning dependencies remain optional feature groups.
- README now positions results as screening evidence requiring expert review.

## [0.1.0] - 2026-10-22

- Initial Sentinel-2 export, preparation, U-Net training, prediction, change detection,
  mapping, validation, synthetic smoke test, and Rondônia pilot report.

[Unreleased]: https://github.com/veerendrakosuri/deforestation-detection/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/veerendrakosuri/deforestation-detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/veerendrakosuri/deforestation-detection/releases/tag/v0.1.0
