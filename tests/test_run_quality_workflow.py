import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_quality_workflow as runner_module
from src.workflows import QualityWorkflowResult


def _fake_workflow_result(
    *,
    status="success",
    final_judgement="accepted",
    stopped_at=None,
) -> QualityWorkflowResult:
    return QualityWorkflowResult(
        workflow_name="phase1_quality_workflow",
        case_id="QW-001",
        status=status,
        final_judgement=final_judgement,
        stopped_at=stopped_at,
        review_reasons=[],
        steps={},
        provider_name="fugu",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        total_latency_seconds=1.0,
        step_latencies={},
    )


def test_cli_trace_flag_writes_jsonl_entry(tmp_path, capsys):
    trace_path = tmp_path / "workflow_runs.jsonl"
    fake_result = _fake_workflow_result()

    with patch.object(runner_module, "run_quality_workflow", return_value=fake_result):
        exit_code = runner_module.main(
            [
                "--provider",
                "fugu",
                "--trace",
                str(trace_path),
            ]
        )

    assert exit_code == 0
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["case_id"] == "QW-001"
    assert entry["workflow_status"] == "success"
    assert entry["provider_name"] == "fugu"
    assert "run_id" in entry and entry["run_id"]
    out = capsys.readouterr().out
    assert "trace appended:" in out


def test_cli_trace_and_output_can_be_used_together(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "result.json"
    fake_result = _fake_workflow_result()

    with patch.object(runner_module, "run_quality_workflow", return_value=fake_result):
        exit_code = runner_module.main(
            [
                "--provider",
                "fugu",
                "--output",
                str(output_path),
                "--trace",
                str(trace_path),
            ]
        )

    assert exit_code == 0
    assert output_path.exists()
    assert trace_path.exists()
    written_output = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_output["case_id"] == "QW-001"
    written_trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert written_trace["case_id"] == "QW-001"


def test_cli_trace_failure_does_not_break_workflow_result(tmp_path, capsys):
    fake_result = _fake_workflow_result()

    # Point --trace at a path whose parent already exists as a *file*. The
    # appender will fail to create a directory there, but the workflow
    # itself must not be affected.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    trace_path = blocker / "nested" / "trace.jsonl"

    with patch.object(runner_module, "run_quality_workflow", return_value=fake_result):
        exit_code = runner_module.main(
            [
                "--provider",
                "fugu",
                "--trace",
                str(trace_path),
            ]
        )

    # Workflow itself was successful, so exit code must still reflect it.
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "WARNING: failed to append trace" in captured.err
