"""CLI smoke tests include failure reports and both invocation styles."""
from pathlib import Path
import json
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("command", ["reference", "simulate", "infer"])
def test_cli_examples(command, tmp_path):
    process = subprocess.run([sys.executable, str(ROOT / "run_model.py"), command,
                              "--steps", "3", "--output", str(tmp_path)], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert "calibrated=false" in process.stdout
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["steps"] == 3 and manifest["data_seed"] == 20260909
    assert not manifest["calibrated"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["first_divergence"] is None
    assert (tmp_path / "trace.jsonl").stat().st_size > 0
    if command == "infer":
        assert summary["oracle_match"] and summary["logical_contexts"] == [8, 4]


def test_cli_module_entry_and_structured_failure(tmp_path):
    # Reject invalid physical ranges and preserve a diagnostic, not a success.
    profile = tmp_path / "analog.json"
    profile.write_text(json.dumps({"reference_code_min": 0, "reference_code_max": 0}))
    process = subprocess.run([sys.executable, "-m", "cluster_model", "simulate", "--steps", "1",
                              "--analog-config", str(profile), "--output", str(tmp_path / "out")],
                             cwd=ROOT, text=True, capture_output=True)
    assert process.returncode == 2
    error = json.loads((tmp_path / "out/error.json").read_text())
    assert error["code"] == "reference_out_of_range"
    assert "candidate_scu" in error and "logical_step" in error
    assert json.loads((tmp_path / "out/summary.json").read_text())["status"] == "failed"


def test_cli_training_is_not_falsely_exposed():
    process = subprocess.run([sys.executable, str(ROOT / "run_model.py"), "train"], text=True, capture_output=True)
    assert process.returncode != 0


@pytest.mark.parametrize("command", ["reference", "simulate", "infer"])
def test_cli_absolute_entry_from_unrelated_directory(command, tmp_path):
    # 重排目录后，旧入口仍须能从项目外运行，且配置支持显式绝对路径。
    output = tmp_path / "result"
    process = subprocess.run(
        [sys.executable, str(ROOT / "run_model.py"), command, "--steps", "1",
         "--config", str(ROOT / "profiles" / "ideal_normalized.json"),
         "--output", str(output)],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["first_divergence"] is None
