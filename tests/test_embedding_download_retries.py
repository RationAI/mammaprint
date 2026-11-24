from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

from ml.data.datasets import _sources


@pytest.mark.parametrize("subdirectory", ["", "nested"])
def test_retry_replaces_partial_file_at_its_original_path(
    tmp_path, monkeypatch, subdirectory
):
    remote = tmp_path / "remote" / "embeddings"
    nested = remote / subdirectory
    nested.mkdir(parents=True)
    (remote / "complete.parquet").write_bytes(b"complete contents")
    (nested / "partial.parquet").write_bytes(b"full parquet contents")
    (nested / "missing.parquet").write_bytes(b"another parquet file")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    uri = remote.as_uri()
    download = LocalArtifactRepository.download_artifacts
    requested = []

    def interrupt_first_download(artifact_uri, dst_path):
        assert artifact_uri == uri
        destination = Path(dst_path) / "embeddings"
        (destination / subdirectory).mkdir(parents=True)
        (destination / "complete.parquet").write_bytes(b"complete contents")
        (destination / subdirectory / "partial.parquet").write_bytes(b"truncated")
        raise MlflowException("interrupted directory download")

    def track_download(self, artifact_path, dst_path=None):
        requested.append(artifact_path)
        return download(self, artifact_path, dst_path)

    monkeypatch.setattr(_sources, "_download_artifacts", interrupt_first_download)
    monkeypatch.setattr(LocalArtifactRepository, "download_artifacts", track_download)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(_sources.tempfile, "tempdir", None)

    destination = _sources.download_artifacts_with_retries(
        uri, initial_backoff_seconds=0
    )
    assert (
        destination / subdirectory / "partial.parquet"
    ).read_bytes() == b"full parquet contents"
    assert (
        destination / subdirectory / "missing.parquet"
    ).read_bytes() == b"another parquet file"
    assert (destination / "complete.parquet").read_bytes() == b"complete contents"
    assert not (destination.parent / "partial.parquet").exists()
    assert set(requested) == {
        (Path("embeddings") / subdirectory / "partial.parquet").as_posix(),
        (Path("embeddings") / subdirectory / "missing.parquet").as_posix(),
    }
