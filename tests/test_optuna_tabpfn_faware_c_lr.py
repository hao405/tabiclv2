from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "optuna_tabpfn_faware_c_lr.py"


def load_module():
    module_name = "module_optuna_tabpfn_faware_c_lr"
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


class FakeFloatTrial:
    def __init__(self, expected_name: str, value: float):
        self.expected_name = expected_name
        self.value = value
        self.params: dict[str, float] = {}
        self.calls: list[tuple[str, float, float, bool]] = []

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        assert name == self.expected_name
        assert low <= self.value <= high
        self.params[name] = self.value
        self.calls.append((name, low, high, log))
        return self.value


class FakeTrial:
    def __init__(self):
        self.number = 0
        self.params: dict[str, float] = {}
        self.user_attrs: dict[str, object] = {}
        self.state = "RUNNING"
        self.value = None

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        if name == "ttt_lr":
            assert low == search.LR_LOW
            assert high == search.LR_HIGH
            assert log is True
            value = 7e-6
        elif name == "ttt_c_reserve_ratio":
            assert low == search.RESERVE_RATIO_LOW
            assert high == search.RESERVE_RATIO_HIGH
            assert log is False
            value = 0.15
        else:
            raise AssertionError(f"unexpected parameter: {name}")
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
                self.best_trial = trial
            except catch as exc:
                trial.state = "FAIL"
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            self.trials.append(trial)


class FakeSampler:
    def __init__(self, seed: int):
        self.seed = seed


class FakeOptuna:
    class samplers:
        TPESampler = FakeSampler

    @staticmethod
    def create_study(study_name: str, direction: str, sampler):
        assert study_name == "demo_study"
        assert direction == "maximize"
        assert isinstance(sampler, FakeSampler)
        assert sampler.seed == 123
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
    (out_dir / "summary.txt").write_text(
        "avg_accuracy_ok: 0.800000\n",
        encoding="utf-8",
    )


def test_joint_sampling_uses_requested_distributions():
    lr_trial = FakeFloatTrial("ttt_lr", 7e-6)
    reserve_trial = FakeFloatTrial("ttt_c_reserve_ratio", 0.15)

    lr = search.sample_ttt_lr(lr_trial)
    reserve_ratio = search.sample_ttt_c_reserve_ratio(reserve_trial)

    assert lr == 7e-6
    assert reserve_ratio == 0.15
    assert lr_trial.calls == [("ttt_lr", search.LR_LOW, search.LR_HIGH, True)]
    assert reserve_trial.calls == [
        (
            "ttt_c_reserve_ratio",
            search.RESERVE_RATIO_LOW,
            search.RESERVE_RATIO_HIGH,
            False,
        )
    ]


def test_trial_path_contains_lr_and_reserve_ratio(tmp_path):
    paths = search.trial_paths(tmp_path, 3, 7e-6, 0.15)

    assert paths.out_dir.parent == tmp_path / "trials"
    assert paths.out_dir.name == "trial_0003_lr_7em06_rr_0p15"
    assert paths.log_path == (
        tmp_path / "logs" / "trial_0003_lr_7em06_rr_0p15.log"
    )


def test_trial_path_rounds_long_sampled_values_for_readability(tmp_path):
    paths = search.trial_paths(
        tmp_path,
        0,
        6.482131165247731e-6,
        0.3802857225639664,
    )

    assert paths.out_dir.name == "trial_0000_lr_6p48em06_rr_0p38"


def test_create_indexed_output_root_uses_next_available_number(tmp_path):
    method_root = tmp_path / "optuna_lr"
    prefix = "tabpfn_v3_optuna_faware_c_lr_reserve_ratio"

    first = search.create_indexed_output_root(method_root, prefix)
    second = search.create_indexed_output_root(method_root, prefix)

    assert first.name == f"{prefix}_1"
    assert second.name == f"{prefix}_2"


def test_default_output_root_uses_numbered_run_names(tmp_path, monkeypatch):
    data_root = tmp_path / "data184"
    data_root.mkdir()
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("print('runner')\n", encoding="utf-8")
    monkeypatch.setattr(search, "REPO_ROOT", tmp_path)
    common_args = [
        "--data-root",
        str(data_root),
        "--runner-path",
        str(runner_path),
        "--n-trials",
        "1",
        "--dry-run",
    ]

    assert search.main(common_args) == 0
    assert search.main(common_args) == 0

    method_root = tmp_path / "results" / "tabpfn" / "v3" / "optuna_lr"
    assert {path.name for path in method_root.iterdir()} == {
        "tabpfn_v3_optuna_faware_c_lr_reserve_ratio_1",
        "tabpfn_v3_optuna_faware_c_lr_reserve_ratio_2",
    }


def test_build_trial_command_contains_joint_parameters(tmp_path):
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
        ttt_lr=7e-6,
        ttt_epochs=30,
        ttt_query_ratio=0.4,
        ttt_c_selection="f_test_centroid_reserve",
        ttt_c_metric="standardized_l2",
        ttt_c_reserve_ratio=0.15,
        ttt_weight_decay=0.01,
        ttt_patience=8,
        ttt_min_delta=1e-4,
        ttt_validation_fraction=0.1,
        ttt_n_estimators_finetune=2,
        ttt_validation_n_estimators=2,
        ttt_grad_accumulation_steps=1,
    )

    assert command[:2] == ["python", str(search.RUNNER_PATH)]
    assert command[command.index("--ttt-lr") + 1] == "7e-06"
    assert command[command.index("--ttt-query-ratio") + 1] == "0.4"
    assert command[command.index("--ttt-c-reserve-ratio") + 1] == "0.15"
    assert command[command.index("--ttt-eval-metric") + 1] == "acc"
    assert command[command.index("--gpus") + 1] == "0"
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--model-version") + 1] == "v2"
    assert command[command.index("--max-datasets") + 1] == "5"
    assert command[command.index("--out-dir") + 1] == str(out_dir)


def test_validate_search_space_rejects_invalid_reserve_ranges():
    with pytest.raises(ValueError, match="reserve ratio bounds"):
        search.validate_search_space(5e-6, 1e-5, 0.6, 0.2, 0.2, "random")

    with pytest.raises(
        ValueError,
        match="must be < 1 for f_test_centroid_reserve",
    ):
        search.validate_search_space(
            5e-6,
            1e-5,
            0.01,
            0.4,
            0.6,
            "f_test_centroid_reserve",
        )

    search.validate_search_space(5e-6, 1e-5, 0.01, 0.4, 0.6, "random")


def test_parser_defaults_use_v3_test_centroid_reserve_and_fixed_query_ratio():
    args = search.build_arg_parser().parse_args([])

    assert args.model_version == "v3"
    assert args.study_name == "tabpfnv3_lr_reserve_ratio"
    assert args.ttt_c_selection == "f_test_centroid_reserve"
    assert args.ttt_query_ratio == 0.2
    assert args.reserve_ratio_low == search.RESERVE_RATIO_LOW
    assert args.reserve_ratio_high == search.RESERVE_RATIO_HIGH


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
    log_path = tmp_path / "logs" / "cached.log"
    write_trial_outputs(out_dir, accuracies={"alpha": 0.8})
    log_path.parent.mkdir(parents=True)
    log_path.write_text("existing log\n", encoding="utf-8")

    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be called when cache is valid")

    reused = search.ensure_trial_outputs(
        out_dir=out_dir,
        log_path=log_path,
        command=["python", "runner.py"],
        runner=fail_runner,
    )

    assert reused is True
    assert log_path.read_text(encoding="utf-8") == "existing log\n"


def test_ensure_trial_outputs_combines_stdout_and_stderr_in_log(tmp_path):
    out_dir = tmp_path / "trial"
    log_path = tmp_path / "logs" / "trial.log"

    def fake_runner(command, cwd, env, check, stdout, stderr):
        assert stderr is subprocess.STDOUT
        stdout.write("runner stdout\n")
        stdout.write("runner stderr\n")
        write_trial_outputs(out_dir, accuracies={"alpha": 0.8})
        return subprocess.CompletedProcess(command, 0)

    reused = search.ensure_trial_outputs(
        out_dir=out_dir,
        log_path=log_path,
        command=["python", "runner.py"],
        runner=fake_runner,
    )

    assert reused is False
    assert log_path.read_text(encoding="utf-8") == (
        "runner stdout\nrunner stderr\n"
    )


def test_dry_run_writes_only_trials_and_best_summary(tmp_path):
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
    assert {path.name for path in output_root.iterdir()} == {
        "trials",
        "logs",
        "best_summary.json",
    }
    assert list((output_root / "logs").iterdir()) == []
    summary = json.loads((output_root / "best_summary.json").read_text())
    assert summary["mode"] == "dry_run"
    assert summary["planned_trials"] == 2
    assert summary["search_space"]["ttt_c_reserve_ratio"] == {
        "low": search.RESERVE_RATIO_LOW,
        "high": search.RESERVE_RATIO_HIGH,
        "log": False,
    }
    for trial in summary["trials"]:
        assert (
            search.RESERVE_RATIO_LOW
            <= trial["ttt_c_reserve_ratio"]
            <= search.RESERVE_RATIO_HIGH
        )
        assert "--ttt-lr" in trial["command"]
        assert "--ttt-query-ratio" in trial["command"]
        assert "--ttt-c-reserve-ratio" in trial["command"]
        assert "_rr_" in Path(trial["out_dir"]).name
        assert Path(trial["log_path"]).parent == output_root / "logs"
        assert Path(trial["log_path"]).stem == Path(trial["out_dir"]).name


def test_main_writes_only_trials_and_best_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "data184"
    data_root.mkdir()
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("print('runner')\n", encoding="utf-8")
    output_root = tmp_path / "optuna_out"

    monkeypatch.setattr(search, "load_optuna", lambda: FakeOptuna)

    def fake_runner(command, cwd, env, check, stdout, stderr):
        out_dir = Path(command[command.index("--out-dir") + 1])
        assert float(command[command.index("--ttt-lr") + 1]) == 7e-6
        assert float(command[command.index("--ttt-query-ratio") + 1]) == 0.2
        assert float(command[command.index("--ttt-c-reserve-ratio") + 1]) == 0.15
        assert command[command.index("--ttt-eval-metric") + 1] == "acc"
        assert stderr is subprocess.STDOUT
        stdout.write("fake trial output\n")
        write_trial_outputs(out_dir, accuracies={"alpha": 0.81, "beta": 0.83})
        return subprocess.CompletedProcess(command, 0)

    original_ensure = search.ensure_trial_outputs

    def fake_ensure_trial_outputs(
        *, out_dir: Path, log_path: Path, command, cwd, force=False, runner=None
    ):
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
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output_root.iterdir()} == {
        "trials",
        "logs",
        "best_summary.json",
    }
    summary = json.loads((output_root / "best_summary.json").read_text())
    assert summary["mode"] == "optuna"
    assert summary["study_name"] == "demo_study"
    assert summary["completed_trials"] == 1
    assert summary["failed_trials"] == 0
    best = summary["best_trial"]
    assert abs(best["objective"] - 0.82) < 1e-12
    assert best["ttt_lr"] == 7e-6
    assert best["ttt_c_reserve_ratio"] == 0.15
    assert best["ok_count"] == 2
    assert Path(best["out_dir"]).name == "trial_0000_lr_7em06_rr_0p15"
    assert Path(best["log_path"]).name == "trial_0000_lr_7em06_rr_0p15.log"
    assert Path(best["log_path"]).read_text(encoding="utf-8") == (
        "fake trial output\n"
    )
