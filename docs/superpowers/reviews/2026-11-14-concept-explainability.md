# Concept explainability: implementation verification and review

Implementation base: `e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2`.
Feature branch: `feat/concept-explainability`.
Numerical reference: IBD `a6fe08b93219b91a68758a4eaf32159a33f5d4eb`.

## Independent review

A separate fresh-context agent reviewed the staged whole feature before commit.
It independently ran the then-current 23 concept tests. Its initial verdict was
request changes: three important issues, no critical issues. The author corrected
all three with regression tests observed failing before the fixes and passing
afterward; the reviewer was not asked to repeat its review.

1. **Coordinate frame and scanner level.** Model docstrings suggested level-0
   coordinates, but the production preprocessing and pinned reader actually use
   selected-level coordinates and choose the scanner level by closest MPP.
   The report now uses that same closest-level rule, scales coordinates only
   when calling the level-0 OpenSlide reader, assembles masks in their stored
   frame, and computes viewport ratios in one consistent frame. Tests cover a
   dataset-key/scanner-level mismatch, non-unit downsampling, exact read location,
   white alpha compositing and overlapping masks. Original model inputs are not
   reinterpreted or changed.
2. **Inherited MLflow run context.** `start_run()` could resume a source training
   run through `MLFLOW_RUN_ID`, even with no active fluent run. The CLI now
   rejects that inherited ID before accessing source artifacts or starting any
   explanation run. A regression test verifies this early failure.
3. **Parameter-free activation mismatch.** Strict weight loading cannot detect
   a different Softmax/Sigmoid module. The raw-logit/index adapter now rejects
   non-Identity source output activations explicitly. A configured Softmax
   checkpoint regression verifies rejection before reporting.

The reviewer also noted one redundant metadata write; the final geometry metadata
is now written once. No review finding remains intentionally deferred in the code.
Unavailable real-data/viewer checks below are operational validation, not claims
of reviewer approval for scientific use.

## Completed checks

- **27 concept tests**, including 30 aggregator/head/objective subtests: copied
  upstream AST fingerprints, signed NNLS/decode, training-only fit, corpus roster
  identity, strict checkpoint load/parity, spatial-input preservation, clipping,
  output semantics, form reveal/export, geometry and TIFF overlap readback.
- **46 tests and 30 subtests across the repository** pass with the existing source
  `preprocessing` namespace pre-imported before pytest collection. The original
  `tests/preprocessing` package otherwise shadows the source namespace; this
  pre-existing collection collision was not fixed by changing training files.
- The synthetic end-to-end test creates temporary TIFFs, embeddings and a
  checkpoint; fits a 16-concept fixture dictionary; evaluates the frozen graph;
  generates masks, xOpat session links and the naming HTML; and checks that the
  checkpoint bytes remain unchanged.
- A separate **actual Ray streaming/shuffle smoke test** successfully fits and
  reloads the dictionary on a synthetic 32-tile, 2,560-dimensional corpus. This
  exercises the copied production iterator, not only the in-process fixture
  iterator. It is not a real cohort result.
- Targeted Ruff checks and formatting pass. CLI Hydra configuration loads.
- Existing dependency versions match the training baseline; ReportFast and its
  optional dependencies are added. No model, training entrypoint, data-loader,
  loss, metric, experiment or data-card source was modified.

Local tests used the existing Python 3.12 environment plus isolated temporary
dependency overlays for pytest, native OpenSlide/libvips and the pinned report
library. The original shared virtualenv was not modified. The resolved report
dependency version preserving the original web stack was tested as well.

Broader-test runner used to avoid the baseline namespace collision:

```python
import sys
from pathlib import Path
import preprocessing
import pytest

sys.path.insert(0, str(Path("tests").resolve()))
raise SystemExit(pytest.main([
    "tests", "--import-mode=importlib", "-p", "no:cacheprovider", "-q"
]))
```

## Decisions made during the port

- Implement inline with one independent review and one final commit, rather
  than per-task commits; this follows the requested review-before-commit order.
- Reuse the existing virtualenv read-only for tests; temporary overlays provide
  absent testing/native/report packages. A fresh deployment must still use the
  committed lockfile and its normal native dependencies.
- Add a bounded, visibly marked in-process fixture stream solely for <=4096-tile
  smoke tests. Production retains the copied Ray iterator; fixture outputs are
  explicitly identified and cannot silently become a normal scientific report.
- Use the already available `rationai-masks` overlap assembler and TIFF writer
  instead of upgrading ratiopath from 1.3.1 to IBD's 1.4.2 solely for its newer
  mask API. Presence averaging is retained; final viewer geometry needs its
  real-slide check.
- Cache the original datasets within one source-run invocation to prevent repeat
  downloads. Membership and original row order remain unchanged.
- Reproduce the naming-page behavior from the checked-in IBD report and supplied
  Slack instructions. Exact deployed-template and dashed-annotation parity remain
  unavailable without Adam's generator.

## Remaining checks and questions

No downloaded WSI, parquet, checkpoint or resolved-run configuration was found by
a repository search including ignored files and common WSI extensions. Thus the
following remain for the VPN-enabled cluster pilot:

1. Recover the selected run's actual SHA/resolved config/checkpoint and original
   label mapping; establish real-checkpoint prediction parity. Do not assume
   every sweep run used the current branch tip.
2. Fit the real training-only dictionary, evaluate held-out fidelity, and decide
   whether K=16 adequately represents each model. Poor fidelity is an explicit
   result; no automatic K search or new XAI method is introduced.
3. Measure GPU memory/runtime, especially transformers; open real gallery tiles
   and masks in xOpat and verify physical alignment and artifact serving.
4. Ask Adam for the exact `naming_k16.html` generator/source run and any existing
   arbitrary-MIL adapter. Confirm the breast pathologist's blinded endpoint
   wording: proposed A/B/no effect/unsure, with index effects shown separately.

These limitations are also stated in the user-facing workflow documentation.
