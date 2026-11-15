"""Exercise multiscale job planning and resumption without cluster side effects."""

import io
import shlex
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir

from scripts.ml import run_train_multiscale_sweep as sweep


REVISION = "a" * 40


class MultiscalePlanTest(unittest.TestCase):
    def test_exact_24_unique_jobs_and_commands_compose(self) -> None:
        jobs = sweep._jobs()
        prefix = sweep._prefix(REVISION)
        names = [sweep._job_name(job, prefix) for job in jobs]
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len(set(names)), 24)
        self.assertEqual(Counter(job.seed for job in jobs), {0: 8, 1: 8, 2: 8})
        with initialize_config_dir(
            version_base=None, config_dir=str(sweep.REPO_ROOT / "configs")
        ):
            for job, name in zip(jobs, names, strict=True):
                with self.subTest(experiment=job.experiment, seed=job.seed):
                    self.assertLessEqual(len(name), 60)
                    arguments = shlex.split(sweep._training_command(job, name))
                    config = compose(config_name="ml", overrides=arguments[6:])
                    self.assertEqual(config.seed, job.seed)
                    self.assertEqual(tuple(config.pyramid.level_order), (4, 3, 2))
                    self.assertEqual(config.trainer.devices, 1)
                    self.assertFalse(config.test_after_fit)
                    self.assertEqual(config.metadata.run_name, name)
                    self.assertEqual(
                        config.label_mode,
                        "type" if job.recipe == "mean_bce" else "index",
                    )
                    self.assertEqual(config.logger.tags.multiscale_screen, job.recipe)

    def test_custom_mapping_is_quoted_and_changes_batch_identity(self) -> None:
        job = sweep._jobs()[0]
        default_prefix = sweep._prefix(REVISION)
        mapping = "/mnt/projects/data with spaces/$(do-not-run) `literal`.csv"
        with patch.object(sweep, "DATA_MAPPING", mapping):
            prefix = sweep._prefix(REVISION)
            arguments = shlex.split(
                sweep._training_command(job, sweep._job_name(job, prefix))
            )
            with initialize_config_dir(
                version_base=None, config_dir=str(sweep.REPO_ROOT / "configs")
            ):
                config = compose(config_name="ml", overrides=arguments[6:])
            self.assertEqual(config.dataset.data_mapping, mapping)
        self.assertNotEqual(prefix, default_prefix)
        self.assertNotEqual(default_prefix, sweep._prefix("b" * 40))

    def test_pod_checks_out_exact_source_and_keeps_shared_mounts(self) -> None:
        job = sweep._jobs()[0]
        fake = SimpleNamespace(
            submit_job=Mock(),
            storage=SimpleNamespace(
                secure=SimpleNamespace(DATA="data", PROJECTS="projects")
            ),
        )
        prefix = sweep._prefix(REVISION)
        name = sweep._job_name(job, prefix)
        with (
            patch.dict(sys.modules, {"kube_jobs": fake}),
            patch.object(sweep, "_existing_kubernetes_job_names", return_value={name}),
            redirect_stdout(io.StringIO()),
        ):
            sweep._submit(job, "A40", prefix, REVISION)
        submitted = fake.submit_job.call_args.kwargs
        self.assertEqual(submitted["storage"], ["data", "projects"])
        self.assertIn(f"git checkout --detach {REVISION}", submitted["script"])
        self.assertEqual(submitted["script"][0], "set -eu")
        self.assertIn(sweep.GIT_REF, submitted["script"][1])
        self.assertIn("trainer.devices=1", submitted["script"][-1])

    def test_unconfirmed_submission_raises(self) -> None:
        fake = SimpleNamespace(
            submit_job=Mock(),
            storage=SimpleNamespace(
                secure=SimpleNamespace(DATA="data", PROJECTS="projects")
            ),
        )
        with (
            patch.dict(sys.modules, {"kube_jobs": fake}),
            patch.object(sweep, "_existing_kubernetes_job_names", return_value=set()),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "did not confirm"),
        ):
            sweep._submit(sweep._jobs()[0], "A40", sweep._prefix(REVISION), REVISION)

    def test_dirty_training_files_are_rejected_before_resolving_revision(self) -> None:
        with (
            patch.object(
                sweep.subprocess, "check_output", return_value=" M configs/ml.yaml\n"
            ) as git,
            self.assertRaisesRegex(RuntimeError, "Commit and push"),
        ):
            sweep._source_revision()
        self.assertEqual(git.call_count, 1)

    def test_queue_targets_and_submission_limit(self) -> None:
        jobs = sweep._jobs()
        selected = sweep._select_jobs(jobs, {}, None)
        self.assertEqual(Counter(gpu for _, gpu in selected), {"A40": 2, "H100": 1})
        self.assertEqual(sweep._select_jobs(jobs, {"A40": 2, "H100": 1}, None), [])
        self.assertEqual(len(sweep._select_jobs(jobs, {}, 1)), 1)
        self.assertEqual(
            [gpu for _, gpu in sweep._select_jobs(jobs, {"A40": 1, "H100": 2}, None)],
            ["A40"],
        )


class MultiscaleRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = sweep._jobs()[:3]
        self.prefix = sweep._prefix(REVISION)
        self.stdout = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(patch.object(sweep, "_jobs", return_value=self.jobs))
        self.enterContext(patch.object(sweep, "_validate_experiments"))
        self.enterContext(
            patch.object(sweep, "_source_revision", return_value=REVISION)
        )
        self.dependencies = self.enterContext(
            patch.object(sweep, "_require_submission_dependencies")
        )
        self.kubernetes = self.enterContext(
            patch.object(sweep, "_existing_kubernetes_job_names", return_value=set())
        )
        self.mlflow = self.enterContext(
            patch.object(sweep, "_existing_mlflow_run_names", return_value=set())
        )
        self.waiting = self.enterContext(
            patch.object(sweep, "_waiting_counts", return_value={})
        )
        self.submit = self.enterContext(patch.object(sweep, "_submit"))
        self.sleep = self.enterContext(patch.object(sweep.time, "sleep"))

    def run_main(self, *arguments: str) -> None:
        with patch("sys.argv", ["run_train_multiscale_sweep.py", *arguments]):
            sweep.main()

    def test_dry_run_needs_no_cluster_dependencies_or_connections(self) -> None:
        self.run_main("--dry-run")
        self.dependencies.assert_not_called()
        self.kubernetes.assert_not_called()
        self.mlflow.assert_not_called()
        self.waiting.assert_not_called()
        self.submit.assert_not_called()
        self.sleep.assert_not_called()
        self.assertEqual(self.stdout.getvalue().count("uv run --frozen"), 3)

    def test_resumes_past_both_kubernetes_and_mlflow_runs(self) -> None:
        self.kubernetes.return_value = {sweep._job_name(self.jobs[0], self.prefix)}
        self.mlflow.return_value = {sweep._job_name(self.jobs[1], self.prefix)}
        self.run_main("--max-jobs", "1")
        self.assertEqual(self.submit.call_args.args[0], self.jobs[2])
        self.submit.assert_called_once()
        self.sleep.assert_not_called()

    def test_refreshes_uncertain_submission_without_resubmitting_created_job(
        self,
    ) -> None:
        self.submit.side_effect = [None, RuntimeError("connection lost"), None]
        self.kubernetes.side_effect = [
            set(),
            {sweep._job_name(self.jobs[1], self.prefix)},
        ]
        self.run_main("--max-jobs", "2")
        self.assertEqual(
            [call.args[0] for call in self.submit.call_args_list], self.jobs
        )
        self.sleep.assert_called_once_with(300)

    def test_full_queue_waits_and_exits_when_all_jobs_appear(self) -> None:
        self.waiting.return_value = sweep.WAITING_JOB_TARGETS.copy()
        self.kubernetes.side_effect = [
            set(),
            {sweep._job_name(job, self.prefix) for job in self.jobs},
        ]
        self.run_main("--poll-interval-seconds", "17")
        self.submit.assert_not_called()
        self.sleep.assert_called_once_with(17)

    def test_zero_submission_limit_is_rejected(self) -> None:
        with (
            redirect_stdout(io.StringIO()),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.run_main("--max-jobs", "0")
        self.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
