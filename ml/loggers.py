"""Project MLflow logger with content-aware checkpoint replacement."""

import hashlib
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rationai.mlkit.lightning.loggers import MLFlowLogger as _MLFlowLogger


def _checkpoint_stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MLFlowLogger(_MLFlowLogger):
    """Replace changed checkpoint bytes while retaining mlkit's artifact format.

    Cache only successful uploads on this instance, per run and artifact stem.
    Lightning saves and notifies synchronously: force hashing the announced saved
    path even if file metadata was preserved; stat-cache other retained files.
    The inherited helper copies bytes and writes MLmodel without reserialization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._uploaded_checkpoints: dict[
            tuple[str, str], tuple[Path, tuple[int, int, int, int], str]
        ] = {}

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if not self._log_model:
            return
        # Also retain immediate-mode callbacks: finalize must flush last links,
        # whose creation happens after Lightning's save notification.
        self._checkpoint_callback = checkpoint_callback
        if self._log_model == "all" or (
            self._log_model is True and checkpoint_callback.save_top_k == -1
        ):
            self._scan_and_log_checkpoints(
                checkpoint_callback,
                saved_path=checkpoint_callback._last_checkpoint_saved,
            )

    @rank_zero_only
    def _scan_and_log_checkpoints(
        self,
        checkpoint_callback: ModelCheckpoint,
        *,
        saved_path: str | None = None,
    ) -> None:
        if not self._log_model:
            return
        checkpoints = self._scan_checkpoints(checkpoint_callback)
        run_id = self.run_id
        logged = {
            Path(artifact.path).stem: artifact.path
            for artifact in self.experiment.list_artifacts(run_id, path="checkpoints")
        }
        saved = Path(saved_path).resolve() if saved_path else None
        for stem, filename in checkpoints.items():
            path = Path(filename)
            canonical = path.resolve()
            try:
                signature = _checkpoint_stat(path)
            except FileNotFoundError:
                if saved is not None and canonical == saved:
                    raise
                # A last symlink can briefly dangle while best is replaced.
                continue
            key = (run_id, stem)
            previous = self._uploaded_checkpoints.get(key)
            if (
                previous is not None
                and previous[:2] == (canonical, signature)
                and canonical != saved
                and stem in logged
            ):
                continue
            digest = _checkpoint_sha256(path)
            if (
                previous is None
                or previous[0] != canonical
                or previous[2] != digest
                or stem not in logged
            ):
                try:
                    self._log_checkpoint(stem, filename)
                except Exception:
                    # Finalize has no saved-path hint. An old matching stat must
                    # not hide a failed same-stat replacement on its next scan.
                    self._uploaded_checkpoints.pop(key, None)
                    raise
            # Never advance this record until the whole inherited upload returns.
            self._uploaded_checkpoints[key] = (canonical, signature, digest)

        for stem in set(logged).difference(checkpoints):
            self.experiment._tracking_client._get_artifact_repo(
                run_id
            ).delete_artifacts(logged[stem])
            self._uploaded_checkpoints.pop((run_id, stem), None)
