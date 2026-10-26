# Operations guide

This toolkit is suitable for repeatable forest-change screening when every alert
is reviewed by a qualified person. It is not an unattended enforcement system.

## Container execution

Release tags publish a CPU image to GitHub Container Registry. In PowerShell,
mount a working directory containing the configuration and GeoTIFF inputs:

```powershell
docker run --rm `
  -v "${PWD}:/work" `
  -w /work `
  ghcr.io/veerendrakosuri/deforestation-detection:latest `
  workflow `
  --before-imagery data/raw/before_sentinel2.tif `
  --before-labels data/raw/before_forest_label.tif `
  --after-imagery data/raw/after_sentinel2.tif `
  --after-labels data/raw/after_forest_label.tif `
  --config project.yaml `
  --output outputs/current `
  --hash-inputs
```

Use an immutable version tag such as `0.3.1` for production runs. The `latest`
tag is convenient for evaluation but may change.

## A safe recurring workflow

1. Select comparable seasonal windows and a reviewed AOI.
2. Export aligned imagery and reference labels using one fixed configuration.
3. Run the workflow and retain `workflow_results.json`, `screening_report.md`, the
   configuration, model checkpoint, and input GeoTIFFs or their immutable archive IDs.
4. Review all large or high-impact patches against the source imagery and an
   independent reference.
5. Record each reviewer decision: confirmed, rejected, uncertain, or field-check needed.
6. Escalate only reviewed alerts and keep the original evidence with the decision.

## Scheduling

Use a scheduler only after one manual run succeeds. A scheduled job should fail
closed: a missing image, incomplete export, invalid configuration, or failed model
run must not reuse an older output as though it were current.

Keep each run in a timestamped output directory. Do not overwrite earlier evidence.
Alert downstream users only after the full command exits successfully and the
expected JSON, report, mask, and map files exist.

## Retention and access

- Keep raw satellite exports according to provider terms and organizational policy.
- Restrict write access to configurations, checkpoints, and reviewed results.
- Never put Google credentials, service-account keys, or private locations in Git.
- Hash inputs for audit-sensitive runs and verify hashes before later review.
- Back up reviewer decisions and the exact released container or package version.

## Monitoring quality

Track confirmed, rejected, and uncertain alerts over time. Investigate changes in
cloud cover, seasons, sensors, forest definitions, and geography before interpreting
a drop in quality as a model problem. Revalidate on geographically separate data
before using a checkpoint in a new biome.

## Incident response

If an output appears wrong, stop automated escalation, preserve the affected run,
and record its input hashes and version. Compare the imagery, labels, configuration,
and model metrics with the last known-good run. Security problems should follow the
private reporting process in `SECURITY.md`.
