"""目录整理的回归约束：公共导入、兼容入口及测试报告不能漂移。"""
from importlib import import_module
import json
from pathlib import Path
import subprocess

from cluster_model import Cluster, ClusterConfig, StepResult
from cluster_model import cli
from scripts import run_tests

ROOT = Path(__file__).resolve().parents[2]


def test_public_imports_and_legacy_entry_are_compatible():
    legacy = import_module("cluster_model.model")
    assert legacy.Cluster is Cluster
    assert legacy.ClusterConfig is ClusterConfig
    assert legacy.StepResult is StepResult
    assert import_module("run_tests").main is run_tests.main
    assert run_tests.ROOT == cli.ROOT == ROOT


def test_test_runner_does_not_reuse_stale_success(tmp_path, monkeypatch):
    report = tmp_path / "reports" / "acceptance"
    report.mkdir(parents=True)
    (report / "junit.xml").write_text('<testsuites><testsuite tests="999"/></testsuites>')
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)

    def fail_before_collection(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        assert not (report / "junit.xml").exists()
        return subprocess.CompletedProcess(command, 4, stdout="", stderr="pytest startup failed")

    monkeypatch.setattr(run_tests.subprocess, "run", fail_before_collection)
    assert run_tests.main() == 4
    summary = json.loads((report / "summary.json").read_text())
    assert summary["exit_code"] == 4 and "passed" not in summary
    assert (report / "pytest-output.txt").read_text() == "pytest startup failed"


def test_test_runner_aggregates_current_junit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    report = tmp_path / "reports" / "acceptance"

    def finish(command, **kwargs):
        (report / "junit.xml").write_text(
            '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1" time="0.1"/>'
            '<testsuite tests="2" failures="0" errors="1" skipped="0" time="0.2"/></testsuites>'
        )
        return subprocess.CompletedProcess(command, 1, stdout="current run", stderr="")

    monkeypatch.setattr(run_tests.subprocess, "run", finish)
    assert run_tests.main() == 1
    summary = json.loads((report / "summary.json").read_text())
    assert summary["tests"] == 5
    assert summary["passed"] == 2
    assert summary["failures"] == summary["errors"] == summary["skipped"] == 1
