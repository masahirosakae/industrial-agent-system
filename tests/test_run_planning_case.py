import json

from scripts.run_planning_case import create_case, main, run_case
from src.llm.base import LLMProvider, LLMResponse


class StaticProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self.text, model="test-model", provider="test")


def test_create_case_uses_predefined_case_id():
    agent_input = create_case("drilling")

    assert agent_input.task_id == "planning-case-drilling"
    assert "M8" in agent_input.source.file_path


def test_run_case_uses_provider_and_returns_parsed_result():
    provider = StaticProvider(
        '{"manufacturing_processes":[{"process_type":"drilling","quantity":4}],"findings":[]}'
    )

    output = run_case("ollama", "drilling", provider=provider)

    assert output["provider"] == "ollama"
    assert output["model"] == "unknown"
    assert output["status"] == "success"
    assert output["parse_success"] is True
    assert output["schema_valid"] is True
    assert output["process_count"] == 1
    assert output["result"]["manufacturing_processes"][0]["process_type"] == "drilling"


def test_main_writes_output_json(monkeypatch, tmp_path):
    output_path = tmp_path / "planning-output.json"
    run_log_path = tmp_path / "planning-runs.jsonl"
    monkeypatch.setattr(
        "scripts.run_planning_case.run_case",
        lambda provider_name, case_name: {
            "provider": provider_name,
            "model": "test-model",
            "case_id": f"planning-case-{case_name}",
            "agent_name": "local_llm_process_planning_agent",
            "status": "success",
            "latency_sec": 0.01,
            "parse_success": True,
            "schema_valid": True,
            "process_count": 0,
            "needs_review": False,
            "error_count": 0,
            "result": {"manufacturing_processes": [], "findings": []},
            "errors": [],
            "notes": [],
        },
    )

    exit_code = main(
        [
            "--provider",
            "ollama",
            "--case",
            "unknown",
            "--output",
            str(output_path),
            "--run-log",
            str(run_log_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["provider"] == "ollama"
    run_record = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert run_record["output_path"] == str(output_path)


def test_main_appends_default_run_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "scripts.run_planning_case.run_case",
        lambda provider_name, case_name: {
            "provider": provider_name,
            "model": "test-model",
            "case_id": f"planning-case-{case_name}",
            "agent_name": "local_llm_process_planning_agent",
            "status": "success",
            "latency_sec": 0.01,
            "parse_success": True,
            "schema_valid": True,
            "process_count": 0,
            "needs_review": False,
            "error_count": 0,
            "result": {"manufacturing_processes": [], "findings": []},
            "errors": [],
            "notes": [],
        },
    )

    assert main(["--provider", "ollama", "--case", "unknown"]) == 0
    assert main(["--provider", "ollama", "--case", "unknown"]) == 0

    records = (tmp_path / "logs" / "planning_runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(records) == 2
    assert json.loads(records[0])["provider"] == "ollama"
    assert json.loads(records[0])["latency_sec"] == 0.01


def test_main_fugu_without_environment_reports_configuration_error(
    monkeypatch, capsys
):
    monkeypatch.delenv("FUGU_API_KEY", raising=False)
    monkeypatch.delenv("FUGU_BASE_URL", raising=False)
    monkeypatch.delenv("FUGU_MODEL", raising=False)

    exit_code = main(["--provider", "fugu", "--case", "drilling", "--no-run-log"])

    assert exit_code == 2
    assert "FUGU_API_KEY is not set" in capsys.readouterr().err
