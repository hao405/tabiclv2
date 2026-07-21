from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "tabpfn_optuna_faware_c_reserve_ratio.py"


def load_module():
    module_name = "module_optuna_tabpfn_faware_c_reserve_ratio"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


search = load_module()


class FakeReserveTrial:
    def __init__(self, value: float):
        self.value = value
        self.params: dict[str, float] = {}

    def suggest_float(self, name: str, low: float, high: float) -> float:
        assert name == "ttt_c_reserve_ratio"
        assert low == search.RESERVE_RATIO_LOW
        assert high == search.RESERVE_RATIO_HIGH
        assert low <= self.value <= high
        self.params[name] = self.value
        return self.value


class FakeTrial:
    def __init__(self):
        self.number = 0
        self.params: dict[str, float] = {}
        self.user_attrs: dict[str, object] = {}
        self.state = "RUNNING"
        self.value = None

    def suggest_float(self, name: str, low: float, high: float) -> float:
        assert name == "ttt_c_reserve_ratio"
        assert low == search.RESERVE_RATIO_LOW
        assert high == search.RESERVE_RATIO_HIGH
        value = 0.037421
        self.params[name] = value
        return value

    def set_user_attr(self, key: str, value) -> None:
        self.user_attrs[key] = value


class FakeStudy:
    def __init__(self, study_name: str):
        self.study_name = study_name
        self.trials: list[FakeTrial] = []
        self.best_trial: FakeTrial | None = None

    def optimize(self, objective, n_trials: int, timeout: int | None, catch):
        assert n_trials == 1
        assert timeout is None
        for trial_number in range(n_trials):
            trial = FakeTrial()
            trial.number = trial_number
            try:
                trial.value = objective(trial)
                trial.state = "COMPLETE"
                if self.best_trial is None or float(trial.value) > float(self.best_trial.value):
                    self.best_trial = trial
            except catch as exc:
                trial.state = "FAIL"
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            self.trials.append(trial)


class FakeSampler:
    def __init__(self, seed: int, n_startup_trials: int):
        self.seed = seed
        self.n_startup_trials = n_startup_trials


class FakeOptuna:
    class samplers:
        TPESampler = FakeSampler

    @staticmethod
    def create_study(study_name: str, storage: str, direction: str, sampler, load_if_exists: bool):
        assert study_name == "demo_study"
        assert direction == "maximize"
        assert isinstance(sampler, FakeSampler)
        assert sampler.seed == 123
        assert sampler.n_startup_trials == 1
        assert load_if_exists is True
        assert storage.startswith("sqlite:///")
        return FakeStudy(study_name=study_name)


def write_trial_outputs(
    out_dir: Path,
    *,
    accuracies: dict[str, float],
    failed_count: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"dataset_name": name, "accuracy": accuracy, "status": "ok"}
        for name, accuracy in accuracies.items()
    ]
    rows.extend(
        {"dataset_name": f"failed_{index}", "accuracy": "", "status": "fail"}
        for index in range(failed_count)
    )
    pd.DataFrame(rows).to_csv(out_dir / "all_classification_results.csv", index=False)
    (out_dir / "summary.txt").write_text("avg_accuracy_ok: 0.800000\n", encoding="utf-8")


def test_sample_reserve_ratio_uses_requested_range():
    trial = FakeReserveTrial(0.123)

    value = search.sample_reserve_ratio(trial)

    assert value == 0.123
    assert trial.params["ttt_c_reserve_ratio"] == 0.123


def test_build_trial_command_contains_tabpfn_runner_and_reserve_ratio(tmp_path):
    out_dir = tmp_path / "trial"

    command = search.build_trial_command(
        python_bin="python",
        runner_path=search.RUNNER_PATH,
        data_root=search.REPO_ROOT / "data184",
        out_dir=out_dir,
        workers=1,
        gpus="0",
        model_version="v2",
        model_path=None,
        random_state=7,
        max_datasets=5,
        n_estimators=8,
        reserve_ratio=0.037421,
        ttt_lr=1e-5,
        ttt_epochs=30,
        ttt_query_ratio=0.2,
        ttt_c_selection="f_test_centroid_reserve",
        ttt_c_metric="standardized_l2",
        ttt_weight_decay=0.01,
        ttt_patience=8,
        ttt_min_delta=1e-4,
        ttt_validation_fraction=0.1,
        ttt_n_estimators_finetune=2,
        ttt_validation_n_estimators=2,
        ttt_grad_accumulation_steps=1,
    )

    assert command[:2] == ["python", str(search.RUNNER_PATH)]
    assert command[1].endswith("baseline_compare/TabPFN-main/Tabpfn_1c_ttt_faware_c.py")
    assert command[command.index("--model-version") + 1] == "v2"
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--gpus") + 1] == "0"
    assert command[command.index("--ttt-lr") + 1] == "1e-05"
    assert command[command.index("--ttt-eval-metric") + 1] == "acc"
    assert (
        command[command.index("--ttt-c-selection") + 1]
        == "f_test_centroid_reserve"
    )
    assert command[command.index("--ttt-c-metric") + 1] == "standardized_l2"
    assert command[command.index("--ttt-c-reserve-ratio") + 1] == "0.037421"
    assert command[command.index("--max-datasets") + 1] == "5"
    assert command[command.index("--out-dir") + 1] == str(out_dir)
    assert "--ttt-c-source" not in command


def test_parser_defaults_use_tabpfn_v3_gpu1_and_expected_reserve_range():
    parser = search.build_arg_parser()
    args = parser.parse_args([])

    assert search.RESERVE_RATIO_LOW == 0.0001
    assert search.RESERVE_RATIO_HIGH == 0.3
    assert args.model_version == "v3"
    assert args.workers == 1
    assert args.gpus == "1"
    assert args.ttt_c_selection == "f_test_centroid_reserve"
    assert args.ttt_c_metric == "standardized_l2"
    assert args.n_trials == 4
    assert args.n_startup_trials == 1
    assert args.expected_datasets == 512


def test_evaluate_trial_outputs_maximizes_status_ok_average_accuracy(tmp_path):
    out_dir = tmp_path / "trial"
    write_trial_outputs(out_dir, accuracies={"alpha": 0.8, "beta": 0.9}, failed_count=1)

    metrics = search.evaluate_trial_outputs(out_dir)

    assert metrics["ok_count"] == 2
    assert metrics["failed_count"] == 1
    assert abs(metrics["avg_accuracy_ok"] - 0.85) < 1e-12
    assert abs(metrics["objective"] - 0.85) < 1e-12


def test_ensure_trial_outputs_reuses_valid_cache(tmp_path):
    out_dir = tmp_path / "cached"
    log_path = tmp_path / "cached.log"
    write_trial_outputs(out_dir, accuracies={"alpha": 0.8})

    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be called when cache is valid")

    reused = search.ensure_trial_outputs(
        out_dir=out_dir,
        log_path=log_path,
        command=["python", "runner.py"],
        runner=fail_runner,
    )

    assert reused is True
    assert not log_path.exists()


def test_dry_run_writes_trials_and_summary(tmp_path):
    data_root = tmp_path / "data184"
    data_root.mkdir()
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("print('runner')\n", encoding="utf-8")
    output_root = tmp_path / "dry_run_out"

    exit_code = search.main(
        [
            "--data-root",
            str(data_root),
            "--runner-path",
            str(runner_path),
            "--output-root",
            str(output_root),
            "--n-trials",
            "2",
            "--sampler-seed",
            "123",
            "--max-datasets",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    trials_df = pd.read_csv(output_root / "trials.csv")
    assert len(trials_df) == 2
    assert set(trials_df["state"]) == {"DRY_RUN"}
    assert "ttt_c_reserve_ratio" in trials_df.columns
    assert trials_df["command"].str.contains("--model-version v3").all()
    assert trials_df["command"].str.contains("--gpus 1").all()
    assert trials_df["command"].str.contains(
        "--ttt-c-selection f_test_centroid_reserve"
    ).all()
    assert trials_df["command"].str.contains("--ttt-c-reserve-ratio").all()
    summary_text = (output_root / "study_summary.txt").read_text(encoding="utf-8")
    assert "dry_run_trials: 2" in summary_text
    assert "reserve_ratio_low: 0.0001" in summary_text
    assert "reserve_ratio_high: 0.3" in summary_text


def test_main_writes_study_outputs_with_fake_optuna(tmp_path, monkeypatch):
    data_root = tmp_path / "data184"
    data_root.mkdir()
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("print('runner')\n", encoding="utf-8")
    output_root = tmp_path / "optuna_out"

    monkeypatch.setattr(search, "load_optuna", lambda: FakeOptuna)

    def fake_runner(command, cwd, stdout, stderr, env, check):
        out_dir = Path(command[command.index("--out-dir") + 1])
        reserve_ratio = float(command[command.index("--ttt-c-reserve-ratio") + 1])
        assert abs(reserve_ratio - 0.037421) < 1e-12
        assert command[command.index("--ttt-eval-metric") + 1] == "acc"
        assert command[command.index("--model-version") + 1] == "v3"
        assert command[command.index("--gpus") + 1] == "1"
        write_trial_outputs(out_dir, accuracies={"alpha": 0.81, "beta": 0.83})
        return subprocess.CompletedProcess(command, 0)

    original_ensure = search.ensure_trial_outputs

    def fake_ensure_trial_outputs(*, out_dir: Path, log_path: Path, command, cwd, force=False, runner=None):
        return original_ensure(
            out_dir=out_dir,
            log_path=log_path,
            command=command,
            cwd=cwd,
            force=force,
            runner=fake_runner,
        )

    monkeypatch.setattr(search, "ensure_trial_outputs", fake_ensure_trial_outputs)

    exit_code = search.main(
        [
            "--data-root",
            str(data_root),
            "--runner-path",
            str(runner_path),
            "--output-root",
            str(output_root),
            "--study-name",
            "demo_study",
            "--n-trials",
            "1",
            "--sampler-seed",
            "123",
            "--max-datasets",
            "5",
            "--expected-datasets",
            "2",
        ]
    )

    assert exit_code == 0
    trials_df = pd.read_csv(output_root / "trials.csv")
    assert len(trials_df) == 1
    assert trials_df.loc[0, "state"] == "COMPLETE"
    assert abs(float(trials_df.loc[0, "value"]) - 0.82) < 1e-12
    assert abs(float(trials_df.loc[0, "ttt_c_reserve_ratio"]) - 0.037421) < 1e-12
    assert int(trials_df.loc[0, "ok_count"]) == 2
    best_params = json.loads((output_root / "best_params.json").read_text(encoding="utf-8"))
    assert abs(best_params["ttt_c_reserve_ratio"] - 0.037421) < 1e-12
    best_summary = json.loads((output_root / "best_trial_summary.json").read_text(encoding="utf-8"))
    assert abs(best_summary["objective"] - 0.82) < 1e-12
    summary_text = (output_root / "study_summary.txt").read_text(encoding="utf-8")
    assert "study_name: demo_study" in summary_text
    assert "completed_trials: 1" in summary_text
    assert "best_ttt_c_reserve_ratio: 0.037421" in summary_text
