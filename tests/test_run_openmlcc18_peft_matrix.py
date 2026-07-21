from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "run_openmlcc18_peft_matrix.sh"
SMOKE_DATASETS = ("OpenML-ID-1063", "OpenML-ID-23381", "OpenML-ID-188")


def _write_python(path: Path, body: str) -> Path:
    path.write_text(
        "import os\nimport shlex\nimport sys\n"
        "with open(os.environ['CALLS_FILE'], 'a', encoding='utf-8') as stream:\n"
        f"    stream.write({path.name!r} + ' ' + shlex.join(sys.argv[1:]) + '\\n')\n"
        + body,
        encoding="utf-8",
    )
    return path


def _make_view(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    view = tmp_path / "view"
    view.mkdir()
    included = []
    for index, name in enumerate(SMOKE_DATASETS):
        dataset = source / name
        dataset.mkdir()
        (dataset / "info.json").write_text("{}\n", encoding="utf-8")
        (view / name).symlink_to(dataset, target_is_directory=True)
        included.append({"dataset_name": name, "n_classes": 2 + index})
    manifest = {
        "source_root": str(source),
        "effective_root": str(view),
        "max_classes": 10,
        "source_count": 72,
        "included_count": 67,
        "excluded_count": 5,
        "included": included,
        "excluded": [],
    }
    (view / "dataset_view_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return view


def _environment(tmp_path: Path, *, validator_exit: int = 0) -> tuple[dict[str, str], Path]:
    calls = tmp_path / "calls.txt"
    validator = _write_python(
        tmp_path / "validator",
        f"raise SystemExit({validator_exit})\n",
    )
    runner = _write_python(
        tmp_path / "runner",
        "raise SystemExit(int(os.environ.get('RUNNER_EXIT', '0')))\n",
    )
    auditor = _write_python(
        tmp_path / "auditor",
        "key = 'SMOKE_AUDIT_EXIT' if '--require-all-success' in sys.argv else 'FULL_AUDIT_EXIT'\n"
        "raise SystemExit(int(os.environ.get(key, '0')))\n",
    )
    view = _make_view(tmp_path)
    env = {
        **os.environ,
        "PEFT_RUNNER": str(runner),
        "VIEW_VALIDATOR": str(validator),
        "MATRIX_AUDITOR": str(auditor),
        "DATA_ROOT": str(view),
        "SMOKE_DATA_ROOT": str(tmp_path / "smoke-view"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "SMOKE_RUN_NAME": "openmlcc18_peft_2x3_smoke_test",
        "FULL_RUN_NAME": "openmlcc18_peft_2x3_full_test",
        "CALLS_FILE": str(calls),
    }
    return env, calls


def _run(tmp_path: Path, preset: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env, _ = _environment(tmp_path)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(WRAPPER), preset],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _parsed_calls(calls: Path) -> list[list[str]]:
    return [shlex.split(line) for line in calls.read_text(encoding="utf-8").splitlines()]


def test_smoke_then_full_builds_separate_fixed_matrix_commands(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(WRAPPER), "smoke-then-full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = _parsed_calls(calls)
    assert [call[0] for call in parsed] == [
        "validator",
        "runner",
        "auditor",
        "runner",
        "auditor",
    ]
    smoke_runner, full_runner = parsed[1], parsed[3]
    for command in (smoke_runner, full_runner):
        assert command[command.index("--model-family") + 1] == "all"
        assert command[command.index("--ttt-peft-method") + 1] == "all"
        assert command[command.index("--workers") + 1] == "2"
        assert command[command.index("--gpu-groups") + 1] == "1;2"
    assert smoke_runner[smoke_runner.index("--run-name") + 1].endswith("_smoke_test")
    assert full_runner[full_runner.index("--run-name") + 1].endswith("_full_test")
    assert "--n-estimators" in smoke_runner
    assert "--n-estimators" not in full_runner
    assert "--matrix-fail-fast" in smoke_runner
    assert "--require-all-peft-success" in smoke_runner
    assert "--require-all-success" in parsed[2]
    assert "--require-all-success" not in parsed[4]


def test_dry_run_validates_but_does_not_launch_or_audit(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path)
    env["DRY_RUN"] = "1"
    completed = subprocess.run(
        ["bash", str(WRAPPER), "smoke-then-full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert [call[0] for call in _parsed_calls(calls)] == ["validator"]
    assert "1\\;2" in completed.stdout
    assert "openmlcc18_peft_2x3_smoke_test" in completed.stdout
    assert "openmlcc18_peft_2x3_full_test" in completed.stdout


def test_validation_failure_blocks_smoke_and_full(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path, validator_exit=7)
    completed = subprocess.run(
        ["bash", str(WRAPPER), "smoke-then-full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 7
    assert [call[0] for call in _parsed_calls(calls)] == ["validator"]


def test_failed_smoke_audit_blocks_full(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path)
    env["SMOKE_AUDIT_EXIT"] = "9"
    completed = subprocess.run(
        ["bash", str(WRAPPER), "smoke-then-full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 9
    assert [call[0] for call in _parsed_calls(calls)] == [
        "validator",
        "runner",
        "auditor",
    ]
    assert "smoke_gate: passed" not in completed.stdout


def test_failed_smoke_runner_is_audited_and_blocks_full(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path)
    env["RUNNER_EXIT"] = "8"
    completed = subprocess.run(
        ["bash", str(WRAPPER), "smoke-then-full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 8
    assert [call[0] for call in _parsed_calls(calls)] == [
        "validator",
        "runner",
        "auditor",
    ]
    assert "smoke_gate: passed" not in completed.stdout


def test_resume_is_forwarded_for_explicit_reused_names(tmp_path: Path) -> None:
    env, calls = _environment(tmp_path)
    env["RESUME"] = "1"
    completed = subprocess.run(
        ["bash", str(WRAPPER), "full"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    runner_call = next(call for call in _parsed_calls(calls) if call[0] == "runner")
    assert "--resume" in runner_call
