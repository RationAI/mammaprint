# Frozen RationAI report worker

This is the existing CPU report implementation used by the successful job
`mammaprint-explain-cpu-config1-20261117` and MLflow report
`fdb0eb59d9194b8e98c88f5ae8dc8a31`. It includes the native tile-removal,
single-tile and attention methods and the standard RationAI HTML assembler.
No numerical method was newly implemented for this snapshot.

The original Kubernetes job carried a source archive with SHA256
`a9cb7019473837cf5a09c61d4a8469a5bb8146e83c10eebd5e7f0727f989fdc6`.
Unused notebooks, tests, launchers and preprocessing sources were omitted here;
every retained file is byte-identical. `manifest.json` lists all checksums.
The original repository licence is retained inside the archive.

`scripts/ml/run_attention_report.py` validates those checksums and changes only
`prepared/plan.json` to bind a selected model/config/checkpoint and fresh output
directory. Dependencies remain frozen by the snapshot's own lockfile; the main
repository training environment is unchanged. The pod template retains only
ConfigMap/Secret references, secure mounts and the previous CPU allocation.

The archive is source code, not model weights or clinical data. It is stored
durably here because the former worker existed only in an expiring job payload.
