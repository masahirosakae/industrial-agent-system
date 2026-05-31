import json

from scripts.run_model_comparison import build_parser, main, run_comparison


def create_success_output(provider_name: str, case_name: str) -> dict:
    return {
        "provider": provider_name,
        "model": "test-model",
        "case_id": f"planning-case-{case_name}",
        "agent_name": "local_llm_process_planning_agent",
        "status": "success",
        "latency_sec": 0.01,
        "parse_success": True,
        "schema_valid": True,
        "process_count": 1,
        "needs_review": False,
        "error_count": 0,
        "errors": [],
    }


def test_run_comparison_logs_each_provider_and_case(monkeypatch, tmp_path):
    log_path = tmp_path / "comparison.jsonl"
    monkeypatch.setattr(
        "scripts.run_model_comparison.run_case",
        lambda provider_name, case_name: create_success_output(provider_name, case_name),
    )

    outputs = run_comparison(
        providers=["ollama", "fugu"],
        cases=["drilling", "unknown"],
        log_path=log_path,
    )

    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(outputs) == 4
    assert len(records) == 4
    assert {(record["provider"], record["case_id"]) for record in records} == {
        ("ollama", "planning-case-drilling"),
        ("ollama", "planning-case-unknown"),
        ("fugu", "planning-case-drilling"),
        ("fugu", "planning-case-unknown"),
    }


def test_run_comparison_logs_failure_and_continues(monkeypatch, tmp_path):
    log_path = tmp_path / "comparison.jsonl"

    def fake_run_case(provider_name, case_name):
        if case_name == "drilling":
            raise ValueError("provider initialization failed")
        return create_success_output(provider_name, case_name)

    monkeypatch.setattr("scripts.run_model_comparison.run_case", fake_run_case)

    outputs = run_comparison(
        providers=["fugu"],
        cases=["drilling", "unknown"],
        log_path=log_path,
    )

    assert [output["status"] for output in outputs] == ["failure", "success"]
    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["errors"] == ["provider initialization failed"]
    assert records[1]["status"] == "success"


def test_main_prints_compact_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "scripts.run_model_comparison.run_case",
        lambda provider_name, case_name: create_success_output(provider_name, case_name),
    )

    exit_code = main(
        [
            "--providers",
            "ollama",
            "--cases",
            "drilling",
            "--run-log",
            str(tmp_path / "comparison.jsonl"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "provider" in stdout
    assert "planning-case-drilling" in stdout
    assert "parse_success" in stdout
    assert "latency_sec" in stdout


def test_parser_accepts_extended_cases():
    args = build_parser().parse_args(
        [
            "--providers",
            "ollama",
            "--cases",
            "tapping,reaming,surface_grinding,turning,mixed_process",
        ]
    )

    assert args.cases == [
        "tapping",
        "reaming",
        "surface_grinding",
        "turning",
        "mixed_process",
    ]
