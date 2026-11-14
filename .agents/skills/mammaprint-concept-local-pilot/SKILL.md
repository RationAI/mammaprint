---
name: mammaprint-concept-local-pilot
description: Use when testing MammaPrint concept explainability locally with cached real embedding bags and bounded downloads of existing MRXS slides from MLflow and WSI service v3.
---

# Local real-asset concept pilot

Run `python scripts/ml/run_local_concept_pilot.py --help` from the repository.
Two complete train bags fit a K16 **smoke-only** fixture dictionary; two complete
validation bags evaluate the frozen checkpoint. This cannot establish
patient-separated validation or whole-cohort fidelity.

## Prerequisites and inputs

Use the locked explainability environment, OpenSlide and libvips. Remote access
requires the lab VPN and authorization for existing-asset reads/local execution.
The established endpoints use VPN access without an application token. For
another deployment use its approved authentication mechanism; never embed tokens
in service URLs or committed files. Preparing scripts grants no execution authority.

Prepare a private JSON list outside Git: exactly two `train` and two `val` entries,
each with `slide_id`, original `source_uri`
(`mlflow-artifacts:/<experiment>/<run>/artifacts/<directory>/<slide>.parquet`),
`local_path`, `sha256`, `bytes`, and preferably inspected `rows`. Complete parquets
must already be cached (<=32 MiB each). Select by size independently of outcomes.
The runner verifies membership, not the original size ranking, and checks every
cached parquet hash before reuse. It retains all rows and coordinates.

Supply the original resolved source config, trusted frozen checkpoint and
persisted original tiling `slides.parquet`, all with independently recovered
SHA256 pins, plus the latter's original artifact URI. Metadata supplies labels
and actual geometry; its `id` must not become an inferred patient identifier.
The mutable training CSV path does not prove historical label bytes. Fetch the
exact recorded training Git object if absent; retain compatibility checks.

## Plan and execute

```bash
python scripts/ml/run_local_concept_pilot.py \
  --tracking-uri https://mlflow.example.org \
  --wsi-base-url https://slides.example.org/wsi-service/v3 \
  --source-run-id SOURCE_RUN --expected-experiment-id 3 \
  --expected-experiment-name MammaPrint \
  --assets-manifest /private/pilot/assets.json \
  --source-config /private/pilot/config-resolved.yaml --source-config-sha256 CONFIG_SHA256 \
  --checkpoint /private/pilot/checkpoint.ckpt --checkpoint-sha256 CHECKPOINT_SHA256 \
  --slides /private/pilot/slides.parquet --slides-sha256 SLIDES_SHA256 \
  --slides-uri mlflow-artifacts:/3/TILING_RUN/artifacts/TILING_DIRECTORY/slides.parquet \
  --output-root /private/pilot/results-v1 --cache-root /private/pilot/cache-v1
```

Replace placeholders with recovered values. Without `--execute`, this reads only
the small roster and prints a plan: no directories, network requests or model
imports. Append `--execute` to the reviewed command for authorized execution.
Use `--skip-wsi --execute` for explicitly numerical-only smoke scope. Use fresh
output/cache roots per attempt, preserving previous evidence. Both are private
(0700) and must be outside the repository. Do not commit clinical artifacts.

Only GET reads are made: MLflow `/api/2.0/mlflow/runs/get`, `/experiments/get`,
`/artifacts/list`; WSI `GET /slides/download?slide_id=...` relative to `/v3`.
The finished run ID, exact training SHA, experiment ID and name are checked.
Original train/val/test artifact **membership metadata** establish slide
disjointness. No test bag is downloaded, scored, invented or selected from.
The adapter calls existing fit/evaluate/report functions with cached train/val
datasets and unchanged original config URI identity. Do not use the full CLI
with original remote embedding directory URIs: it downloads whole splits.

## Bounds and result evidence

At most two validation WSIs are requested. Defaults: 512 MiB/archive and 1 GiB
combined compressed; 2 GiB/archive and 4 GiB combined expanded. Override only with
explicit bounded scope using `--archive-byte-cap`, `--total-byte-cap`,
`--expanded-byte-cap`, `--total-expanded-byte-cap` (integer bytes).
Unknown Content-Length is streamed within the strict remaining budget. Reaching
the cap without confirmed EOF fails closed, even if the ZIP ends exactly there.
Aborted bytes count against total budget and partial files are removed.
Extraction rejects traversal, absolute/drive paths, symlinks, collisions and
expanded-size overflow; Windows separators and MRXS sidecar structure survive.

A unique expected MRXS, Index.dat and Slidedat.ini are required. OpenSlide must
open it and recovered geometry must match original tiling metadata. With fewer
than two complete WSIs, numerical validation continues and `report_status`
records why no report was built. A cropped TIFF or lone `.mrxs` is not a substitute.

Inspect `pilot_manifest.json`, dictionary/validation `manifest.json`, and, when
built, `validation/report_manifest.json` plus `naming_k16.html`. These record
source hashes, selected original membership, subset fixture scope, caps, WSI
outcomes, frozen CPU execution and unverified patient separation. Coefficients
retain complete bag rows; evaluation tables expose reconstruction/model effects.
Two-slide SD and gallery shortages cannot establish scientific validity. Local
report creation does not establish remote xOpat access to local masks or live
viewer alignment. Nothing is uploaded or logged to MLflow.

## Diagnosis, cleanup and safe verification

Hash/identity/provenance/geometry mismatches stop execution: investigate the real
discrepancy; do not replace the recorded source SHA or invent split/patient data.
For network errors check VPN and explicit targets. `ByteCapExceededError` means
the complete-WSI attempt hit its fixed bound: preserve evidence and accept
numerical-only scope or obtain authorization for a larger bounded attempt.
Do not repeatedly download to evade combined caps.

No remote rollback is needed: no source run mutation, upload, source-model
training or cluster submission occurs. After inspecting results, remove only the
explicit local output/cache roots to clean up; original cached inputs stay intact.

```bash
python -m unittest tests.test_local_concept_pilot -v
```

Safe tests cover plan isolation, cache hashes, strict transfer budgets/cleanup,
ZIP safety, target identity, truthful split provenance and complete local bag
loading. They do not establish live service availability, real-checkpoint parity
or clinical validity; record actual authorized execution evidence separately.
