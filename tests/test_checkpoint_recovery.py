"""Bounded recovery tests use metadata fakes and tiny local CPU models only."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def core():
    from ml import recover_checkpoint

    return recover_checkpoint


def config():
    return {
        "seed": 0,
        "mode": "fit",
        "checkpoint": None,
        "test_after_fit": True,
        "ml": {"_target_": "ml.models.MammaprintModule"},
        "datamodule": {"_target_": "ml.data.DataModule"},
        "logger": {"_target_": "rationai.mlkit.lightning.loggers.MLFlowLogger"},
        "trainer": {
            "_target_": "rationai.mlkit.lightning.Trainer",
            "max_epochs": 10,
            "callbacks": {
                "best": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "monitor": "valid/loss",
                    "mode": "min",
                    "save_top_k": 1,
                }
            },
        },
    }


def event(value, step, timestamp):
    return SimpleNamespace(value=value, step=step, timestamp=timestamp)


def evidence(cfg=None, monitor=None, epochs=None):
    c = core()
    cfg = cfg or config()
    raw = json.dumps(cfg).encode()
    return c.SourceEvidence(
        c.SourceIdentity("https://tracking.example", "a" * 32, "3"),
        "3",
        "b" * 40,
        raw,
        cfg,
        monitor or [event(0.7, 3, 100), event(0.2, 7, 200), event(0.2, 11, 300)],
        epochs or [event(0, 3, 100), event(1, 7, 200), event(2, 11, 300)],
        {"records": 6, "bytes": 100},
    )


def test_help_is_stdlib_only():
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run(
        [sys.executable, "-S", "-m", "ml.recover_checkpoint", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "--execute" in p.stdout and "--publish" in p.stdout


def test_identity_url_and_explicit_tracking_agree():
    c = core()
    source = c.parse_source(
        "https://tracking.example/#/experiments/3/runs/" + "a" * 32,
        "https://tracking.example/",
    )
    assert source.run_id == "a" * 32 and source.experiment_id == "3"
    for source, uri in [
        ("https://other.example/#/experiments/3/runs/abc", "https://tracking.example"),
        ("abc", "https://user:secret@tracking.example"),
        ("abc\n", "https://tracking.example"),
        ("abc", "https://tracking.example?token=secret"),
        ("${evil}", "https://tracking.example"),
    ]:
        with pytest.raises(ValueError):
            c.parse_source(source, uri)


def test_plan_earliest_tie_preserves_original_horizon():
    c = core()
    e = evidence()
    original = copy.deepcopy(e.config)
    p = c.plan_replay(e, implementation_sha="c" * 40)
    assert p["stop_epoch"] == 1 and p["completed_epoch_cap"] == 2
    assert p["target_score"] == 0.2 and p["mode"] == "min"
    assert p["trainer_max_epochs"] == 10
    assert e.config == original
    assert p["source_config_sha256"] != ""


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e.monitor_history.append(event(float("nan"), 15, 400)), "finite"),
        (lambda e: e.monitor_history.append(event(0.1, 7, 201)), "multiple|conflict"),
        (lambda e: e.epoch_history.append(event(9, 7, 200)), "ambiguous"),
        (lambda e: e.config["trainer"].update(min_epochs=3), "minimum"),
        (lambda e: e.config["trainer"].update(val_check_interval=0.5), "cadence"),
        (lambda e: e.config.update(checkpoint="old.ckpt"), "initialization"),
        (lambda e: e.config["trainer"].update(min_steps=20), "minimum"),
        (lambda e: e.config["ml"].update(_target_="os.system"), "target"),
        (
            lambda e: e.config["trainer"]["callbacks"]["best"].update(
                every_n_train_steps=2
            ),
            "cadence",
        ),
    ],
)
def test_plan_rejects_unsafe_or_ambiguous_source(mutation, match):
    e = evidence()
    mutation(e)
    with pytest.raises(ValueError, match=match):
        core().plan_replay(e, implementation_sha="c" * 40)


def test_epoch_alignment_fallback_and_checkpoint_eligibility():
    c = core()
    e = evidence(epochs=[event(0, 3, 99), event(1, 7, 199), event(2, 11, 299)])
    e.config["trainer"]["callbacks"]["best"]["every_n_epochs"] = 2
    e.monitor_history[0].value = 0.01
    p = c.plan_replay(e, implementation_sha="c" * 40)
    assert p["stop_epoch"] == 1 and p["epoch_alignment"] == "unambiguous_step"
    e.epoch_history.append(event(3, 7, 198))
    with pytest.raises(ValueError, match="ambiguous"):
        c.plan_replay(e, implementation_sha="c" * 40)


def test_explicit_selection_is_validation_only_and_recorded():
    c = core()
    p = c.plan_replay(
        evidence(), implementation_sha="c" * 40, monitor="valid/f1", mode="max"
    )
    assert p["selection_changed"] is True and p["stop_epoch"] == 0
    with pytest.raises(ValueError, match="validation"):
        c.plan_replay(
            evidence(), implementation_sha="c" * 40, monitor="test/f1", mode="max"
        )
    with pytest.raises(ValueError):
        c.plan_replay(evidence(), implementation_sha="short")


class Pages(list):
    def __init__(self, values, token=None):
        super().__init__(values)
        self.token = token


class Store:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        self.calls.append((run_id, metric_key, max_results, page_token))
        return self.pages[page_token]


def test_history_reader_pages_completely_and_rejects_truncation():
    c = core()
    store = Store(
        {None: Pages([event(1, 1, 1)], "next"), "next": Pages([event(2, 2, 2)])}
    )
    reader = c.PagedMetricReader(store, c.MetadataLimits(page_size=1, max_records=10))
    assert len(reader.read("a" * 32, "epoch")) == 2
    assert store.calls[-1][-1] == "next"
    store.pages["next"].token = "next"
    with pytest.raises(ValueError, match="token"):
        c.PagedMetricReader(store, c.MetadataLimits(page_size=1)).read(
            "a" * 32, "epoch"
        )
    with pytest.raises(ValueError, match="limit"):
        c.PagedMetricReader(store, c.MetadataLimits(page_size=1, max_records=1)).read(
            "a" * 32, "epoch"
        )


class MetadataClient:
    def __init__(self, cfg=None, experiment="3", sha="b" * 40):
        self.cfg = json.dumps(cfg or config()).encode()
        self.run = SimpleNamespace(
            info=SimpleNamespace(
                run_id="a" * 32,
                experiment_id=experiment,
                status="FINISHED",
                lifecycle_stage="active",
            ),
            data=SimpleNamespace(tags={"mlflow.source.git.commit": sha}),
        )
        self.calls = []

    def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return self.run

    def list_artifacts(self, run_id, path):
        self.calls.append(("list", path))
        return [
            SimpleNamespace(
                path="configs/config-resolved.yaml",
                is_dir=False,
                file_size=len(self.cfg),
            )
        ]

    def download_artifacts(self, run_id, artifact_path, dst_path):
        self.calls.append(("download", artifact_path))
        path = Path(dst_path) / "config-resolved.yaml"
        path.write_bytes(self.cfg)
        return str(path)


def test_metadata_plan_has_only_bounded_reads(tmp_path):
    c = core()
    client = MetadataClient()
    store = Store({None: Pages([event(0, 3, 100)])})
    reader = c.PagedMetricReader(store, c.MetadataLimits())
    e = c.read_source(
        client, c.parse_source("a" * 32, "https://tracking.example"), reader=reader
    )
    assert e.source_training_sha == "b" * 40
    assert all(call[0] in {"get_run", "list", "download"} for call in client.calls)
    client.run.info.experiment_id = "4"
    with pytest.raises(ValueError, match="experiment"):
        c.read_source(
            client,
            c.SourceIdentity("https://tracking.example", "a" * 32, "3"),
            reader=reader,
        )
    client.run.info.experiment_id = "3"
    with pytest.raises(ValueError, match="conflict"):
        c.read_source(
            client,
            c.parse_source("a" * 32, "https://tracking.example"),
            reader=reader,
            training_sha="d" * 40,
        )
    with pytest.raises(ValueError, match="limit"):
        c.read_source(
            client,
            c.parse_source("a" * 32, "https://tracking.example"),
            reader=reader,
            limits=c.MetadataLimits(config_bytes=1),
        )


def test_effective_config_changes_only_operational_allowlist(tmp_path):
    c = core()
    e = evidence()
    original = copy.deepcopy(e.config)
    plan = c.plan_replay(e, implementation_sha="c" * 40)
    effective, changes = c.effective_config(e.config, plan, tmp_path, "d" * 32)
    assert e.config == original
    assert (
        effective["ml"] == original["ml"]
        and effective["datamodule"] == original["datamodule"]
    )
    assert effective["trainer"]["max_epochs"] == 10
    assert effective["logger"]["run_id"] == "d" * 32
    assert effective["test_after_fit"] is False
    assert changes
    effective["ml"]["bad"] = 1
    with pytest.raises(ValueError, match="allowlist"):
        c.validate_effective_config(original, effective, plan)


def test_native_budget_preserves_scheduler_prefix_and_best_checkpoint(tmp_path):
    import torch
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    c = core()

    class Tiny(LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.lrs = []

        def training_step(self, batch, batch_idx):
            self.lrs.append(self.trainer.optimizers[0].param_groups[0]["lr"])
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("valid/loss", float(5 - self.current_epoch))

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]

    loader = DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=2)

    def run(stop, root):
        model = Tiny()
        checkpoint = ModelCheckpoint(
            dirpath=root, monitor="valid/loss", mode="min", filename="best"
        )
        trainer = Trainer(
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint]
            + ([] if stop is None else [c.ReplayBudgetCallback(stop)]),
        )
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        return (
            model,
            trainer,
            torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=False
            ),
        )

    full, _, _ = run(None, tmp_path / "full")
    bounded, trainer, checkpoint = run(1, tmp_path / "bounded")
    assert bounded.lrs == full.lrs[:2]
    assert (
        trainer.max_epochs == 5 and checkpoint["epoch"] == 1 and len(bounded.lrs) == 2
    )
    first, _, checkpoint = run(0, tmp_path / "first")
    assert len(first.lrs) == 1 and checkpoint["epoch"] == 0


def test_budget_ignores_sanity_and_target_miss_is_explicit():
    c = core()
    callback = c.ReplayBudgetCallback(0)
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=True, should_stop=False)
    callback.on_train_epoch_end(trainer, None)
    assert trainer.should_stop is False
    trainer.sanity_checking = False
    callback.on_train_epoch_end(trainer, None)
    assert trainer.should_stop is True
    plan = c.plan_replay(evidence(), implementation_sha="c" * 40)
    manifest = c.result_manifest(
        plan,
        recovery_run_id="d" * 32,
        score=0.5,
        checkpoint_epoch=1,
        checkpoint_global_step=8,
        checkpoint_sha256="e" * 64,
        checkpoint_size=10,
    )
    assert manifest["target_reached"] is False and manifest["status"] == "target_missed"
    assert manifest["historical_test_metrics_revalidated"] is False
    with pytest.raises(ValueError, match="finite"):
        c.result_manifest(
            plan,
            recovery_run_id="d" * 32,
            score=float("inf"),
            checkpoint_epoch=1,
            checkpoint_global_step=8,
            checkpoint_sha256="e" * 64,
            checkpoint_size=10,
        )


def test_inherited_mlflow_context_fails_before_mutations(monkeypatch, tmp_path):
    c = core()
    monkeypatch.setenv("MLFLOW_RUN_ID", "a" * 32)

    class NoWrites:
        def __getattr__(self, name):
            raise AssertionError("Must reject context before accessing client")

    with pytest.raises(ValueError, match="MLFLOW_RUN_ID"):
        c.execute_replay(
            evidence(),
            c.plan_replay(evidence(), implementation_sha="c" * 40),
            tmp_path,
            client=NoWrites(),
        )


def test_original_multiline_description_is_safe_metadata():
    e = evidence()
    e.config["metadata"] = {"description": "original training\nnotes"}
    assert core().plan_replay(e, implementation_sha="c" * 40)["stop_epoch"] == 1


def test_dependency_closure_allows_extras_but_keeps_original_transitive_pins():
    c = core()
    project = {"project": {"dependencies": ["lightning>=2"]}}
    packages = [
        {
            "name": "lightning",
            "version": "2.5",
            "source": {"registry": "https://pypi.org/simple"},
            "dependencies": [{"name": "torch"}],
        },
        {
            "name": "torch",
            "version": "2.8",
            "source": {"registry": "https://pypi.org/simple"},
        },
    ]
    old = c._dependency_closure(project, {"package": packages})
    packages.append(
        {
            "name": "explainability-extra",
            "version": "1",
            "source": {"registry": "https://pypi.org/simple"},
        }
    )
    assert c._dependency_closure(project, {"package": packages}) == old
    packages[1]["version"] = "2.9"
    assert c._dependency_closure(project, {"package": packages}) != old


def test_training_ast_allows_only_logger_annotation_import():
    c = core()
    old = "from rationai.mlkit.lightning.loggers import MLFlowLogger\ndef train():\n    fit()\n"
    fixed = old.replace("rationai.mlkit.lightning.loggers", "ml.loggers")
    assert c._training_ast(old) == c._training_ast(fixed)
    assert c._training_ast(old) != c._training_ast(fixed.replace("fit()", "test()"))


@pytest.mark.parametrize(
    "score,expected_status", [(0.1, "completed"), (0.5, "target_missed")]
)
def test_execute_native_fit_only_fresh_same_experiment_and_verified_artifact(
    monkeypatch, tmp_path, score, expected_status
):
    import hashlib
    import shutil

    import hydra.utils
    import torch
    import yaml
    from lightning.pytorch import LightningDataModule, LightningModule, Trainer
    from lightning.pytorch.loggers import Logger
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, TensorDataset

    c = core()
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

    class Client(MetadataClient):
        def __init__(self):
            super().__init__()
            self.writes, self.tags, self.status = [], {}, None

        def create_run(self, experiment_id, tags, run_name):
            assert experiment_id == "3"
            self.tags = tags
            self.writes.append(("create", "d" * 32))
            return SimpleNamespace(
                info=SimpleNamespace(run_id="d" * 32, experiment_id="3")
            )

        def log_artifact(self, run_id, path, artifact_path):
            assert run_id == "d" * 32
            destination = tmp_path / "remote" / artifact_path / Path(path).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            self.writes.append(("artifact", run_id))

        def download_artifacts(self, run_id, path, directory):
            assert run_id == "d" * 32
            return str(tmp_path / "remote" / path)

        def set_tag(self, run_id, key, value):
            assert run_id == "d" * 32
            self.tags[key] = value

        def set_terminated(self, run_id, status):
            assert run_id == "d" * 32
            self.status = status

    client = Client()
    stages = []

    class Data(LightningDataModule):
        def setup(self, stage=None):
            stages.append(stage)
            assert stage == "fit"

        def train_dataloader(self):
            return DataLoader(TensorDataset(torch.ones(1, 2)))

        def val_dataloader(self):
            return self.train_dataloader()

        def test_dataloader(self):
            raise AssertionError("Must never touch test bags")

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("valid/loss", score)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    class LocalLogger(Logger):
        @property
        def experiment(self):
            return client

        name = "local"
        version = "1"

        def log_hyperparams(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            assert not any(key.startswith("test/") for key in metrics)

        def _log_checkpoint(self, key, path):
            client.log_artifact("d" * 32, path, "unused")
            directory = tmp_path / "remote" / "checkpoints" / key
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, directory / "checkpoint.ckpt")
            (directory / "MLmodel").write_text(
                yaml.safe_dump(
                    {
                        "run_id": "d" * 32,
                        "model_size_bytes": Path(path).stat().st_size,
                        "flavors": {"pytorch": {"model_data": "checkpoint.ckpt"}},
                    }
                )
            )

    native_instantiate = hydra.utils.instantiate
    seen = []

    def instantiate(node, **kwargs):
        target = node["_target_"]
        seen.append(target)
        if target == "ml.models.MammaprintModule":
            return Model()
        if target == "ml.data.DataModule":
            assert kwargs == {"_recursive_": False}
            return Data()
        if target == "ml.loggers.MLFlowLogger":
            assert (
                node["run_id"] == "d" * 32
                and node["tracking_uri"] == "https://tracking.example"
            )
            return LocalLogger()
        if target == "rationai.mlkit.lightning.Trainer":
            callbacks = [
                native_instantiate(value) for value in node["callbacks"].values()
            ]
            return Trainer(
                max_epochs=node["max_epochs"],
                accelerator="cpu",
                devices=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=kwargs["logger"],
                callbacks=callbacks,
                default_root_dir=node["default_root_dir"],
            )
        return native_instantiate(node, **kwargs)

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(c, "ensure_training_revision", lambda sha: None)
    monkeypatch.setattr(
        c,
        "verify_replay_provenance",
        lambda *a, **k: {"historical_data_hashes_verified": False},
    )
    e = evidence()
    original = copy.deepcopy(e.config)
    manifest = c.execute_replay(
        e,
        c.plan_replay(e, implementation_sha="c" * 40),
        tmp_path / "output",
        client=client,
    )
    assert manifest["status"] == expected_status and client.status == "FINISHED"
    assert stages == ["fit"] and e.config == original
    assert client.tags["recovery.source_run_id"] == "a" * 32
    assert (
        client.tags["recovery.source_config_sha256"] == manifest["source_config_sha256"]
    )
    assert (
        hashlib.sha256((tmp_path / "remote" / c.CHECKPOINT).read_bytes()).hexdigest()
        == manifest["checkpoint_sha256"]
    )
    effective = OmegaConf.to_container(
        OmegaConf.load(Path(manifest["output_directory"]) / "config-effective.yaml")
    )
    assert effective["trainer"]["max_epochs"] == 10
    assert not effective["test_after_fit"] and effective["ml"] == original["ml"]
    assert (
        tmp_path / "remote" / "recovery" / "config-original.yaml"
    ).read_bytes() == e.config_bytes


def test_config_aliases_and_unresolved_interpolation_are_rejected():
    c = core()
    with pytest.raises(ValueError, match="aliases"):
        c.parse_config_bytes(b"x: &loop {self: *loop}")
    cfg = config()
    cfg["ml"]["foo"] = "${oc.env:SECRET}"
    with pytest.raises(ValueError, match="interpolation"):
        c.parse_config_bytes(json.dumps(cfg).encode())


def test_provenance_gate_checks_runtime_and_original_pins(monkeypatch, tmp_path):
    import ml.concept_explainability.model as model

    c = core()
    root = tmp_path / "repo"
    (root / "ml").mkdir(parents=True)
    source = "from ml.loggers import MLFlowLogger\ndef train():\n    fit()\n"
    (root / "ml" / "train.py").write_text(source)
    project = '[project]\nrequires-python = ">=3.12,<3.13"\ndependencies = ["demo"]\n'
    lock = 'requires-python = "==3.12.*"\n[[package]]\nname = "demo"\nversion = "1.0"\nsource = {registry = "https://pypi.org/simple"}\n'
    (root / "uv.lock").write_text(lock)
    historical = {"pyproject.toml": project, "uv.lock": lock, "ml/train.py": source}

    def git(*args, **kwargs):
        if args[0] == "rev-parse":
            return "c" * 40
        if args[0] in {"status", "cat-file", "diff"}:
            return ""
        assert args[0] == "show"
        return historical[args[1].split(":", 1)[1]]

    monkeypatch.setattr(c, "_git", git)
    scientific_checks = []
    monkeypatch.setattr(
        model,
        "verify_training_revision",
        lambda sha, **k: scientific_checks.append(sha),
    )
    monkeypatch.setattr(
        c.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="1.0"),
    )
    plan = c.plan_replay(evidence(), implementation_sha="c" * 40)
    report = c.verify_replay_provenance(plan, config(), repository=root)
    assert (
        scientific_checks == ["b" * 40]
        and report["historical_runtime_verified"] is False
    )
    (root / "uv.lock").write_text(lock.replace('version = "1.0"', 'version = "2.0"'))
    with pytest.raises(ValueError, match="closure"):
        c.verify_replay_provenance(plan, config(), repository=root)
    (root / "uv.lock").write_text(lock)
    monkeypatch.setattr(
        c.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="2.0"),
    )
    with pytest.raises(ValueError, match="Installed"):
        c.verify_replay_provenance(plan, config(), repository=root)


def test_active_mlflow_run_rejected(monkeypatch):
    import mlflow

    c = core()
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setattr(
        mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="a" * 32)),
    )
    with pytest.raises(ValueError, match="active MLflow"):
        c._reject_inherited_context()


def test_publication_cli_dispatches_without_planning_or_training(
    monkeypatch, tmp_path, capsys
):
    import mlflow

    import ml.checkpoint_publication as publication

    c = core()
    manifest = {
        "source_run_id": "a" * 32,
        "experiment_id": "3",
        "tracking_uri": "https://tracking.example",
        "implementation_sha": "c" * 40,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    client = object()
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **k: client)
    calls = []
    monkeypatch.setattr(
        publication,
        "publish_replacement",
        lambda *args: calls.append(args) or {"status": "published"},
    )
    monkeypatch.setattr(
        c, "read_source", lambda *a, **k: pytest.fail("Publish must not plan/retrain")
    )
    assert (
        c.main(
            [
                "a" * 32,
                "--tracking-uri",
                "https://tracking.example",
                "--implementation-sha",
                "c" * 40,
                "--publish",
                "--replace-original",
                "--manifest",
                str(path),
                "--expected-old-sha256",
                "e" * 64,
            ]
        )
        == 0
    )
    assert calls == [(client, manifest, "e" * 64)]
    assert json.loads(capsys.readouterr().out) == {"status": "published"}


@pytest.mark.parametrize("view", ["model-metrics", "artifacts", "overview"])
def test_sheet_run_view_links_are_accepted(view):
    identity = core().parse_source(
        "https://tracking.example/#/experiments/3/runs/" + "a" * 32 + "/" + view,
        "https://tracking.example",
    )
    assert identity.run_id == "a" * 32 and identity.experiment_id == "3"


@pytest.mark.parametrize("reached", [True, False])
def test_one_job_replay_then_replace_pins_old_bytes_and_blocks_missed_target(
    monkeypatch, tmp_path, reached
):
    import ml.checkpoint_publication as publication

    c = core()
    sequence = []
    manifest = {
        "target_reached": reached,
        "output_directory": str(tmp_path),
        "recovery_run_id": "d" * 32,
    }
    monkeypatch.setattr(c, "_reject_inherited_context", lambda: None)
    monkeypatch.setattr(
        publication,
        "read_checkpoint_hash",
        lambda *a: sequence.append("pin") or "e" * 64,
    )
    monkeypatch.setattr(
        c, "execute_replay", lambda *a, **k: sequence.append("fit") or manifest
    )

    def publish(client, result, old):
        assert result is manifest and old == "e" * 64
        sequence.append("publish")
        return {"status": "completed"}

    monkeypatch.setattr(publication, "publish_replacement", publish)
    client = SimpleNamespace(log_artifact=lambda *a: sequence.append("receipt"))
    e = evidence()
    plan = c.plan_replay(e, implementation_sha="c" * 40)
    if reached:
        result = c.recover_and_replace(
            e, plan, tmp_path, client=client, replace_original=True
        )
        assert result["publication"]["status"] == "completed"
        assert sequence == ["pin", "fit", "publish", "receipt"]
    else:
        with pytest.raises(ValueError, match="missed"):
            c.recover_and_replace(
                e, plan, tmp_path, client=client, replace_original=True
            )
        assert sequence == ["pin", "fit"]
    sequence.clear()
    with pytest.raises(ValueError, match="SHA256"):
        c.recover_and_replace(
            e,
            plan,
            tmp_path,
            client=client,
            replace_original=True,
            expected_old_sha256="f" * 64,
        )
    assert sequence == ["pin"]


def test_recorded_target_stops_early_only_at_eligible_checkpoint_and_minimums():
    c = core()
    callback = c.ReplayBudgetCallback(19, "valid/loss", "min", 0.2, every_n_epochs=2)
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=4,
        min_steps=0,
        min_epochs=0,
        sanity_checking=False,
        should_stop=False,
        callback_metrics={"valid/loss": 0.1},
    )
    callback.on_train_epoch_end(trainer, None)
    assert not trainer.should_stop
    trainer.current_epoch = 1
    trainer.min_epochs = 3
    callback.on_train_epoch_end(trainer, None)
    assert not trainer.should_stop
    trainer.min_epochs = 0
    callback.on_train_epoch_end(trainer, None)
    assert trainer.should_stop


def test_missing_training_object_fetches_only_verified_origin(monkeypatch):
    c = core()
    calls = []
    present = False

    def git(*args, **kwargs):
        nonlocal present
        calls.append(args)
        if args[:2] == ("cat-file", "-e"):
            if not present:
                raise subprocess.CalledProcessError(1, ["git", *args])
        elif args == ("remote", "get-url", "origin"):
            return "https://github.com/RationAI/mammaprint.git\n"
        elif args[0] == "fetch":
            present = True
        return ""

    monkeypatch.setattr(c, "_git", git)
    c.ensure_training_revision("b" * 40)
    assert ("fetch", "--no-tags", "--no-write-fetch-head", "origin", "b" * 40) in calls
    calls.clear()
    c.ensure_training_revision("b" * 40)
    assert calls == [("cat-file", "-e", "b" * 40 + "^{commit}")]

    present = False
    monkeypatch.setattr(
        c,
        "_git",
        lambda *a, **k: (
            git(*a, **k) if a[0] != "remote" else "https://other.example/repo"
        ),
    )
    with pytest.raises(ValueError, match="unexpected origin"):
        c.ensure_training_revision("b" * 40)
