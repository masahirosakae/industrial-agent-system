import json

from scripts.run_quality_issue_comparison import build_parser, main, run_comparison


def create_success_output(provider_name: str, case_name: str) -> dict:
    return {
        "provider": provider_name,
        "model": "test-model",
        "case_id": f"quality-case-{case_name}",
        "agent_name": "quality_issue_analysis_agent",
        "status": "success",
        "needs_review": False,
        "latency_sec": 0.01,
        "parse_success": True,
        "schema_valid": True,
        "error_type": None,
        "cause_count": 2,
        "error_count": 0,
        "raw_output": "{\"ok\": true}",
        "errors": [],
    }


def test_run_comparison_logs_each_provider_and_case(monkeypatch, tmp_path):
    log_path = tmp_path / "qi-comparison.jsonl"
    monkeypatch.setattr(
        "scripts.run_quality_issue_comparison.run_case",
        lambda provider_name, case_name: create_success_output(
            provider_name, case_name
        ),
    )

    outputs = run_comparison(
        providers=["ollama", "fugu"],
        cases=["screw_fastening", "solder_wetting"],
        log_path=log_path,
    )

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(outputs) == 4
    assert len(records) == 4
    assert {(r["provider"], r["case_id"]) for r in records} == {
        ("ollama", "quality-case-screw_fastening"),
        ("ollama", "quality-case-solder_wetting"),
        ("fugu", "quality-case-screw_fastening"),
        ("fugu", "quality-case-solder_wetting"),
    }
    assert all(record["needs_review"] is False for record in records)
    assert all(record["raw_output"] == "{\"ok\": true}" for record in records)


def test_run_comparison_logs_failure_and_continues(monkeypatch, tmp_path):
    log_path = tmp_path / "qi-comparison.jsonl"

    def fake_run_case(provider_name, case_name):
        if case_name == "screw_fastening":
            raise ValueError("provider initialization failed")
        return create_success_output(provider_name, case_name)

    monkeypatch.setattr(
        "scripts.run_quality_issue_comparison.run_case", fake_run_case
    )

    outputs = run_comparison(
        providers=["fugu"],
        cases=["screw_fastening", "solder_wetting"],
        log_path=log_path,
    )

    assert [o["status"] for o in outputs] == ["failure", "success"]
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["errors"] == ["provider initialization failed"]
    assert records[0]["needs_review"] is True
    assert records[0]["error_type"] == "comparison_error"
    assert records[1]["status"] == "success"


def test_main_prints_compact_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "scripts.run_quality_issue_comparison.run_case",
        lambda provider_name, case_name: create_success_output(
            provider_name, case_name
        ),
    )

    exit_code = main(
        [
            "--providers",
            "ollama",
            "--cases",
            "dimensional_variation",
            "--run-log",
            str(tmp_path / "qi-comparison.jsonl"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "provider" in stdout
    assert "quality-case-dimensional_variation" in stdout
    assert "needs_review" in stdout
    assert "error_type" in stdout
    assert "cause_count" in stdout


def test_parser_accepts_all_cases():
    args = build_parser().parse_args(
        [
            "--providers",
            "ollama,fugu",
            "--cases",
            "screw_fastening,solder_wetting,dimensional_variation",
        ]
    )

    assert args.providers == ["ollama", "fugu"]
    assert args.cases == [
        "screw_fastening",
        "solder_wetting",
        "dimensional_variation",
    ]
