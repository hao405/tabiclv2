from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_tabular_seed_search_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("seed_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_matrix_has_25_method_cells_and_exclusions():
    module = load_module()
    matrix = module.full_matrix()
    assert len(matrix) == 25
    assert ("data184", "tabiclv2", "infer") not in matrix
    assert ("data184", "tabiclv2", "ft") not in matrix
    assert ("data184", "tabiclv2", "faware_ft") not in matrix
    assert len([cell for cell in matrix if cell[0] == "data184"]) == 11
    assert len([cell for cell in matrix if cell[0] == "openmlcc18"]) == 14


def test_default_seed_contract():
    module = load_module()
    assert module.DEFAULT_LONG_SEEDS == (
        3,
        10,
        16,
        42,
        2025,
        2026,
        2027,
        2028,
        2029,
        2030,
        2031,
        2032,
        2033,
        2034,
        2035,
    )
    assert module.DEFAULT_SHORT_SEEDS == (3, 10, 16)
    args = module.build_parser().parse_args(["--run-id", "defaults"])
    assert args.long_seeds == module.DEFAULT_LONG_SEEDS
    assert args.short_seeds == module.DEFAULT_SHORT_SEEDS
    assert args.gpus == (0, 1, 2)


def test_mixed_seed_policy_has_9_long_16_short_and_183_slots():
    module = load_module()
    matrix = module.full_matrix()
    long_cells = [
        cell for cell in matrix if module.seed_policy_for_method(cell[2]) == "long"
    ]
    short_cells = [
        cell for cell in matrix if module.seed_policy_for_method(cell[2]) == "short"
    ]
    assert len(long_cells) == 9
    assert len(short_cells) == 16
    assert (
        len(long_cells) * len(module.DEFAULT_LONG_SEEDS)
        + len(short_cells) * len(module.DEFAULT_SHORT_SEEDS)
        == 183
    )


def test_seed_flag_mapping(tmp_path):
    module = load_module()
    identity = {"label": "data184", "count": 184}

    def make_spec(family, method, runner):
        return module.CellSpec(
            cell_id=f"data184__tabiclv2__{method}",
            order=0,
            dataset="data184",
            model="tabiclv2",
            method=method,
            runner_family=family,
            runner=runner,
            data_root="data184",
            dataset_identity=identity,
            config_source="test",
            config_status="ready",
            config_error=None,
            static_args=[],
            config_fingerprint="abc",
            seed_policy=module.seed_policy_for_method(method),
            seeds=[42],
        )

    tfm = module.build_command(
        make_spec("tfm", "infer", str(module.TFM_RUNNER)),
        seed=42,
        gpu=1,
        output_dir=tmp_path / "tfm",
        python_bin="python",
        retry_failed=False,
    )
    peft = module.build_command(
        make_spec("peft", "lora", str(module.PEFT_RUNNER)),
        seed=42,
        gpu=1,
        output_dir=tmp_path / "peft",
        python_bin="python",
        retry_failed=False,
    )
    micp = module.build_command(
        make_spec("micp", "micp", str(module.MICP_RUNNER)),
        seed=42,
        gpu=1,
        output_dir=tmp_path / "micp",
        python_bin="python",
        retry_failed=False,
    )
    assert tfm[tfm.index("--random-state") + 1] == "42"
    assert "--seed" not in tfm
    assert peft[peft.index("--random-state") + 1] == "42"
    assert "--seed" not in peft
    assert micp[micp.index("--seed") + 1] == "42"
    assert "--random-state" not in micp


def test_tabpfn_best_summary_requires_complete_trial(tmp_path):
    module = load_module()
    path = tmp_path / "best_summary.json"
    path.write_text(
        json.dumps(
            {
                "best_trial": {
                    "state": "COMPLETE",
                    "ttt_lr": 2.5e-6,
                    "ttt_c_reserve_ratio": 0.15,
                    "command": (
                        "python runner.py --ttt-lr 2.5e-6 "
                        "--ttt-c-reserve-ratio 0.15 "
                        "--ttt-c-metric standardized_l2 "
                        "--ttt-weight-decay 0.01 "
                        "--ttt-min-delta 0.0001"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    params, source = module.load_tabpfn_best_summary(path)
    assert params == {
        "ttt_lr": 2.5e-6,
        "ttt_c_reserve_ratio": 0.15,
        "ttt_c_metric": "standardized_l2",
        "ttt_weight_decay": 0.01,
        "ttt_min_delta": 0.0001,
        "source_command": (
            "python runner.py --ttt-lr 2.5e-6 "
            "--ttt-c-reserve-ratio 0.15 "
            "--ttt-c-metric standardized_l2 "
            "--ttt-weight-decay 0.01 "
            "--ttt-min-delta 0.0001"
        ),
    }
    assert source == str(path)

    path.write_text(json.dumps({"best_trial": None}), encoding="utf-8")
    params, error = module.load_tabpfn_best_summary(path)
    assert params is None
    assert "best_trial is not complete" in error


def test_tuned_tfm_passthrough_follows_dynamic_launcher_args(tmp_path):
    module = load_module()
    static_args = module.tfm_static_args(
        model="tabpfnv3",
        method="faware_ft",
        best_params={
            "ttt_lr": 5e-6,
            "ttt_c_reserve_ratio": 0.15,
            "ttt_c_metric": "standardized_l2",
            "ttt_weight_decay": 0.01,
            "ttt_min_delta": 0.0001,
            "ttt_eval_metric_runner": "acc",
        },
        smoke=False,
    )
    spec = module.CellSpec(
        cell_id="data184__tabpfnv3__faware_ft",
        order=0,
        dataset="data184",
        model="tabpfnv3",
        method="faware_ft",
        runner_family="tfm",
        runner=str(module.TFM_RUNNER),
        data_root="data184",
        dataset_identity={"label": "data184", "count": 184},
        config_source="best",
        config_status="ready",
        config_error=None,
        static_args=static_args,
        config_fingerprint="expected",
        seed_policy="long",
        seeds=[42],
    )
    command = module.build_command(
        spec,
        seed=42,
        gpu=1,
        output_dir=tmp_path / "out",
        python_bin="python",
        retry_failed=False,
    )
    marker = command.index("--")
    assert command.index("--data-root") < marker
    assert command.index("--out-dir") < marker
    assert command.index("--random-state") < marker
    assert command[command.index("--ttt-c-metric") + 1] == "standardized_l2"
    assert command[marker + 1 :] == [
        "--ttt-weight-decay",
        "0.01",
        "--ttt-min-delta",
        "0.0001",
        "--ttt-eval-metric",
        "acc",
    ]


def test_exact_terminal_reuse_requires_fingerprint(tmp_path):
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "test",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--skip-gpu-wait",
        ]
    )
    spec = module.CellSpec(
        cell_id="data184__tabiclv2__micp",
        order=0,
        dataset="data184",
        model="tabiclv2",
        method="micp",
        runner_family="micp",
        runner=str(module.MICP_RUNNER),
        data_root="data184",
        dataset_identity={"label": "data184", "count": 184},
        config_source="test",
        config_status="ready",
        config_error=None,
        static_args=[],
        config_fingerprint="expected",
        seed_policy="short",
        seeds=[42],
    )
    manager = module.QueueManager(args, [spec], tmp_path / "test")
    output_dir = tmp_path / "test" / "data184" / "tabiclv2" / "micp" / "seed42"
    output_dir.mkdir(parents=True)
    terminal = {
        "dataset": "data184",
        "model": "tabiclv2",
        "method": "micp",
        "seed": 42,
        "status": "success",
        "config_fingerprint": "wrong",
    }
    module.atomic_json(manager.seed_terminal_path(output_dir), terminal)
    state = module.SeedState(seed=42)
    assert not manager.maybe_reuse_seed(spec, state, output_dir)
    terminal["config_fingerprint"] = "expected"
    module.atomic_json(manager.seed_terminal_path(output_dir), terminal)
    assert manager.maybe_reuse_seed(spec, state, output_dir)
    assert state.status == "success"
    assert state.execution == "reused"


def test_gpu_check_falls_back_when_nvidia_smi_is_missing(tmp_path, monkeypatch):
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "test",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--gpu-ready-checks",
            "1",
            "--gpu-poll-seconds",
            "0",
        ]
    )
    spec = module.CellSpec(
        cell_id="data184__tabiclv2__micp",
        order=0,
        dataset="data184",
        model="tabiclv2",
        method="micp",
        runner_family="micp",
        runner=str(module.MICP_RUNNER),
        data_root="data184",
        dataset_identity={"label": "data184", "count": 184},
        config_source="test",
        config_status="ready",
        config_error=None,
        static_args=[],
        config_fingerprint="expected",
        seed_policy="short",
        seeds=[42],
    )
    manager = module.QueueManager(args, [spec], tmp_path / "test")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"free": 95, "total": 100, "name": "test-gpu"}),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    manager.wait_for_gpu(1, "worker0")
    assert calls[0][0] == "nvidia-smi"
    assert calls[1][0] == args.python_bin


def test_prequeue_manifest_registers_cells_without_freezing_blocked_params(tmp_path):
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "test",
            "--output-root",
            str(tmp_path),
        ]
    )
    ready = module.CellSpec(
        cell_id="data184__tabiclv2__micp",
        order=0,
        dataset="data184",
        model="tabiclv2",
        method="micp",
        runner_family="micp",
        runner=str(module.MICP_RUNNER),
        data_root="data184",
        dataset_identity={"label": "data184", "count": 184},
        config_source="formal",
        config_status="ready",
        config_error=None,
        static_args=["--secret-frozen-arg", "value"],
        config_fingerprint="ready-fingerprint",
        seed_policy="short",
        seeds=[3, 10],
    )
    blocked = module.CellSpec(
        **{
            **ready.__dict__,
            "cell_id": "data184__tabpfnv3__faware_ft",
            "model": "tabpfnv3",
            "method": "faware_ft",
            "config_status": "blocked_config",
            "config_error": "best summary missing",
            }
        )
    args.gpus = (1, 2)
    path = module.write_prequeue_manifest(args, [ready, blocked], tmp_path / "test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_for_config"
    assert payload["planned_cells"] == 2
    assert payload["planned_seed_slots"] == 4
    assert payload["cells"][1]["config_status"] == "blocked_config"
    assert "static_args" not in payload["cells"][0]
    assert "config_fingerprint" not in payload["cells"][0]

    args.gpus = (0, 1, 2)
    module.write_prequeue_manifest(args, [ready, blocked], tmp_path / "test")
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["gpus"] == [0, 1, 2]
    assert "resource_update" in updated


def test_shared_coverage_reports_mixed_seed_expectations(tmp_path):
    module = load_module()
    args = module.build_parser().parse_args(
        [
            "--run-id",
            "mixed",
            "--mode",
            "smoke",
            "--output-root",
            str(tmp_path),
        ]
    )
    identity = {"label": "data184", "count": 184}
    long_spec = module.CellSpec(
        cell_id="data184__tabpfnv3__infer",
        order=0,
        dataset="data184",
        model="tabpfnv3",
        method="infer",
        runner_family="tfm",
        runner=str(module.TFM_RUNNER),
        data_root="data184",
        dataset_identity=identity,
        config_source="test",
        config_status="ready",
        config_error=None,
        static_args=[],
        config_fingerprint="long",
        seed_policy="long",
        seeds=[3, 42],
    )
    short_spec = module.CellSpec(
        **{
            **long_spec.__dict__,
            "cell_id": "data184__tabpfnv3__micp",
            "order": 1,
            "method": "micp",
            "runner_family": "micp",
            "runner": str(module.MICP_RUNNER),
            "config_fingerprint": "short",
            "seed_policy": "short",
            "seeds": [3],
        }
    )
    manager = module.QueueManager(args, [long_spec, short_spec], tmp_path / "mixed")
    for spec in (long_spec, short_spec):
        for seed_state in manager.states[spec.cell_id].seeds:
            output_dir = (
                tmp_path
                / "outputs"
                / spec.method
                / f"seed{seed_state.seed}"
            )
            output_dir.mkdir(parents=True)
            (output_dir / "all_classification_results.csv").write_text(
                "dataset_name,status\nshared_dataset,ok\n",
                encoding="utf-8",
            )
            seed_state.output_dir = str(output_dir)

    manager.write_shared_coverage()
    rows = json.loads(
        (tmp_path / "mixed" / "shared_success_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    by_seed = {row["seed"]: row for row in rows}
    assert by_seed[3]["seed_policy"] == "shared_short_long"
    assert by_seed[3]["expected_method_count"] == 2
    assert by_seed[3]["method_count"] == 2
    assert by_seed[42]["seed_policy"] == "long_only"
    assert by_seed[42]["expected_method_count"] == 1
    assert by_seed[42]["method_count"] == 1
