# Security policy

## Supported version

Security fixes are applied to the latest released version.

## Reporting a vulnerability

Do not open a public issue for a vulnerability involving credentials, arbitrary code
execution, dependency compromise, or private geospatial data. Use GitHub's private
vulnerability reporting feature for this repository. If that feature is unavailable,
contact the repository owner through their GitHub profile and request a private channel
without including exploit details.

Reports should include the affected version, impact, reproduction steps, and any known
mitigation. Please allow maintainers reasonable time to investigate before disclosure.

## Credential handling

This project does not need credentials in its source tree. Earth Engine authentication
is handled by Google's client. Never commit OAuth codes, service-account keys, `.env`
files, cloud-project secrets, or downloaded private imagery.
