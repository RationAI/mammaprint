import io
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.ml import run_train_sweep as sweep


def job_name(job: sweep.SweepJob) -> str:
    return sweep._job_name(job[1], job[3], job[7], job[6], job[4], job[8])


class QueueSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = sweep._sweep_jobs()

    def test_empty_queues_fill_to_each_target(self) -> None:
        selected, deferred = sweep._jobs_to_submit(self.jobs, {}, None)
        self.assertEqual(deferred, 0)
        self.assertEqual(
            Counter(resources["gpu"] for _, resources in selected),
            {"A40": 2, "H100": 1, "mig-2g.20gb": 2, "mig-1g.10gb": 2},
        )
        self.assertEqual(len({job_name(job) for job, _ in selected}), 7)

    def test_existing_queues_only_fill_deficits(self) -> None:
        waiting = {"A40": 1, "H100": 3, "mig-2g.20gb": 2, "mig-1g.10gb": 0}
        selected, _ = sweep._jobs_to_submit(self.jobs, waiting, None)
        self.assertEqual(
            Counter(resources["gpu"] for _, resources in selected),
            {"A40": 1, "mig-1g.10gb": 2},
        )
        self.assertEqual(waiting["A40"], 1)

    def test_full_queues_submit_nothing(self) -> None:
        selected, _ = sweep._jobs_to_submit(self.jobs, sweep.WAITING_JOB_TARGETS, None)
        self.assertEqual(selected, [])

    def test_restricted_jobs_wait_for_h100_capacity(self) -> None:
        restricted = [
            job for job in self.jobs if job[3] in (2, 3) and job[6] == "transformer"
        ]
        selected, _ = sweep._jobs_to_submit(restricted, {"H100": 1}, None)
        self.assertEqual(selected, [])
        selected, _ = sweep._jobs_to_submit(restricted, {}, None)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1]["gpu"], "H100")
        self.assertEqual(selected[0][1]["memory"], "56Gi")

    def test_level_four_transformers_do_not_use_mig10(self) -> None:
        restricted = [
            job for job in self.jobs if job[3] == 4 and job[6] == "transformer"
        ]
        selected, _ = sweep._jobs_to_submit(
            restricted, {"A40": 2, "H100": 1, "mig-2g.20gb": 2}, None
        )
        self.assertEqual(selected, [])

    def test_unconfigured_h100_jobs_are_deferred(self) -> None:
        restricted = [
            job for job in self.jobs if job[3] in (2, 3) and job[6] == "transformer"
        ]
        with patch.object(sweep, "GPUS", ("A40",)):
            selected, deferred = sweep._jobs_to_submit(restricted, {}, None)
        self.assertEqual(selected, [])
        self.assertEqual(deferred, len(restricted))

    def test_submission_limit_caps_the_refill(self) -> None:
        selected, _ = sweep._jobs_to_submit(self.jobs, {}, 3)
        self.assertEqual(len(selected), 3)

    def test_resource_and_gpu_policies_for_entire_grid(self) -> None:
        self.assertEqual(len(self.jobs), 11_520)
        counts = dict.fromkeys(sweep.GPUS, 0)
        for job in self.jobs:
            resources = sweep._resources_for_job(counts, *job[1:])
            self.assertIsNotNone(resources)
            assert resources is not None
            self.assertEqual(resources["cpu"], 8)
            if job[3] in (2, 3):
                if job[6] == "transformer":
                    self.assertEqual(resources["gpu"], "H100")
                    self.assertEqual(resources["memory"], "56Gi")
                elif job[6] == "attention":
                    self.assertEqual(resources["memory"], "40Gi")
                else:
                    self.assertEqual(resources["memory"], "20Gi")
            else:
                self.assertEqual(resources["memory"], "16Gi")
                if job[3] == 4 and job[6] == "transformer":
                    self.assertNotEqual(resources["gpu"], "mig-1g.10gb")


class RefillLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = sweep._sweep_jobs()[:3]
        self.stdout = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.jobs_mock = self.enterContext(
            patch.object(sweep, "_sweep_jobs", return_value=self.jobs)
        )
        self.kubernetes = self.enterContext(
            patch.object(sweep, "_existing_kubernetes_job_names", return_value=set())
        )
        self.mlflow = self.enterContext(
            patch.object(sweep, "_existing_mlflow_run_names", return_value=set())
        )
        self.waiting = self.enterContext(patch.object(sweep, "pending_gpu_counts"))
        self.submit = self.enterContext(patch.object(sweep, "_submit_sweep_job"))
        self.sleep = self.enterContext(patch.object(sweep.time, "sleep"))

    def run_main(self, *arguments: str) -> None:
        with patch("sys.argv", ["run_train_sweep.py", *arguments]):
            sweep.main()

    def test_refills_again_and_avoids_duplicates_before_mlflow_appears(self) -> None:
        self.waiting.side_effect = [
            {"A40": 1, "H100": 1, "mig-2g.20gb": 2, "mig-1g.10gb": 2},
            {"A40": 0, "H100": 1, "mig-2g.20gb": 2, "mig-1g.10gb": 2},
        ]
        self.run_main("--max-jobs", "3")
        submitted = [call.args[0] for call in self.submit.call_args_list]
        self.assertEqual(submitted, self.jobs)
        self.assertEqual(self.kubernetes.call_count, 2)
        self.assertEqual(self.mlflow.call_count, 2)
        self.sleep.assert_called_once_with(300)

    def test_refreshes_external_submissions_and_stops_when_complete(self) -> None:
        self.waiting.return_value = sweep.WAITING_JOB_TARGETS.copy()
        self.kubernetes.side_effect = [set(), {job_name(job) for job in self.jobs}]
        self.run_main("--poll-interval-seconds", "17")
        self.submit.assert_not_called()
        self.sleep.assert_called_once_with(17)

    def test_lookup_failure_waits_before_retrying(self) -> None:
        self.mlflow.side_effect = [RuntimeError("MLflow unavailable"), set()]
        self.waiting.return_value = {}
        self.run_main("--max-jobs", "1")
        self.assertEqual(self.submit.call_count, 1)
        self.sleep.assert_called_once_with(300)
        self.assertIn("Queue refill failed", self.stdout.getvalue())

    def test_partial_submission_failure_refreshes_before_retrying(self) -> None:
        self.waiting.return_value = {}
        self.submit.side_effect = [None, RuntimeError("connection lost"), None]
        # The failed call actually created the job; the next poll must skip it.
        self.kubernetes.side_effect = [set(), {job_name(self.jobs[1])}]
        self.run_main("--max-jobs", "2")
        self.assertEqual(
            [call.args[0] for call in self.submit.call_args_list], self.jobs
        )
        self.sleep.assert_called_once_with(300)

    def test_interrupt_stops_cleanly(self) -> None:
        self.waiting.return_value = sweep.WAITING_JOB_TARGETS.copy()
        self.sleep.side_effect = KeyboardInterrupt
        self.run_main()
        self.submit.assert_not_called()
        self.assertIn("Sweep queue refiller stopped", self.stdout.getvalue())


class SubmissionConfirmationTests(unittest.TestCase):
    def test_unconfirmed_submission_raises(self) -> None:
        job = sweep._sweep_jobs()[0]
        with (
            patch.object(sweep, "submit_job") as submit,
            patch.object(sweep, "_existing_kubernetes_job_names", return_value=set()),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "did not confirm"),
        ):
            sweep._submit_sweep_job(job, {"gpu": "A40", "cpu": 8, "memory": "20Gi"})
        submit.assert_called_once()

    def test_confirmed_submission_preserves_training_parameters(self) -> None:
        job = sweep._sweep_jobs()[0]
        with (
            patch.object(sweep, "submit_job") as submit,
            patch.object(
                sweep, "_existing_kubernetes_job_names", return_value={job_name(job)}
            ) as lookup,
            redirect_stdout(io.StringIO()),
        ):
            sweep._submit_sweep_job(job, {"gpu": "A40", "cpu": 8, "memory": "20Gi"})
        lookup.assert_called_once_with({job_name(job)})
        self.assertEqual(submit.call_args.kwargs["job_name"], job_name(job))
        self.assertEqual(submit.call_args.kwargs["gpu"], "A40")
        self.assertIn(
            "data/embedded=tissue_only/l2", submit.call_args.kwargs["script"][-1]
        )


if __name__ == "__main__":
    unittest.main()
