import json

import pytest

from scripts.run_quality_issue_case import (
    QUALITY_ISSUE_CASES,
    append_run_log,
    get_case,
    main,
    run_case,
)
from src.llm.base import LLMProvider, LLMResponse


class StaticProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(text=self.text, model="test-model", provider="test")


class BoomProvider(LLMProvider):
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        raise RuntimeError("provider blew up")


VALID = json.dumps(
    {
        "issue_summary": "ok",
        "suspected_causes": [
            {"cause": "c1", "evidence": "e1", "likelihood": "medium"}
        ],
        "containment_actions": ["a1"],
        "investigation_plan": [
            {"step": "s1", "owner": "qe", "expected_signal": "x"}
        ],
        "additional_data_needed": ["d1"],
        "risk_if_unresolved": "r1",
        "verification_points": ["v1"],
    }
)


@pytest.mark.parametrize(
    "case_name",
    ["screw_fastening", "solder_wetting", "dimensional_variation"],
)
def test_get_case_returns_expected_cases(case_name):
    case = get_case(case_name)
    assert case.case_id == f"quality-case-{case_name}"


def test_run_case_uses_provider_and_returns_parsed_result():
    provider = StaticProvider(VALID)

    output = run_case("ollama", "screw_fastening", provider=provider)

    assert output["provider"] == "ollama"
    assert output["status"] == "success"
    assert output["needs_review"] is False
    assert output["parse_success"] is True
    assert output["schema_valid"] is True
    assert output["error_type"] is None
    assert output["cause_count"] == 1
    assert output["raw_output"] == VALID
    assert output["result"]["issue_summary"] == "ok"


def test_run_case_empty_json_is_not_schema_valid():
    output = run_case("ollama", "screw_fastening", provider=StaticProvider("{}"))

    assert output["status"] == "needs_review"
    assert output["needs_review"] is True
    assert output["parse_success"] is True
    assert output["schema_valid"] is False
    assert output["error_type"] == "schema_validation_error"
    assert output["raw_output"] == "{}"


@pytest.mark.parametrize(
    ("provider", "expected_parse_success", "expected_error_type"),
    [
        (StaticProvider("not JSON"), False, "parse_error"),
        (StaticProvider("{}"), True, "schema_validation_error"),
        (BoomProvider(), False, "provider_error"),
    ],
)
def test_append_run_log_records_failure_classification(
    tmp_path,
    provider,
    expected_parse_success,
    expected_error_type,
):
    log_path = tmp_path / "qi-runs.jsonl"
    output = run_case("ollama", "screw_fastening", provider=provider)

    append_run_log(output, log_path, output_path=None)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["status"] == "needs_review"
    assert record["needs_review"] is True
    assert record["parse_success"] is expected_parse_success
    assert record["schema_valid"] is False
    assert record["error_type"] == expected_error_type


def test_main_writes_output_json(monkeypatch, tmp_path):
    output_path = tmp_path / "qi-output.json"
    run_log_path = tmp_path / "qi-runs.jsonl"

    monkeypatch.setattr(
        "scripts.run_quality_issue_case.run_case",
        lambda provider_name, case_name: {
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
            "cause_count": 1,
            "error_count": 0,
            "result": {key: [] for key in (
                "suspected_causes",
                "containment_actions",
                "investigation_plan",
                "additional_data_needed",
                "verification_points",
            )} | {"issue_summary": "x", "risk_if_unresolved": "r"},
            "raw_output": "{\"issue_summary\":\"x\"}",
            "errors": [],
            "notes": [],
        },
    )

    exit_code = main(
        [
            "--provider",
            "ollama",
            "--case",
            "solder_wetting",
            "--output",
            str(output_path),
            "--run-log",
            str(run_log_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["provider"] == "ollama"
    record = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert record["case_id"] == "quality-case-solder_wetting"
    assert record["needs_review"] is False
    assert record["error_type"] is None
    assert record["raw_output"] == "{\"issue_summary\":\"x\"}"


def test_main_fugu_without_environment_reports_configuration_error(
    monkeypatch, capsys
):
    monkeypatch.delenv("FUGU_API_KEY", raising=False)
    monkeypatch.delenv("FUGU_BASE_URL", raising=False)
    monkeypatch.delenv("FUGU_MODEL", raising=False)

    exit_code = main(
        ["--provider", "fugu", "--case", "screw_fastening", "--no-run-log"]
    )

    assert exit_code == 2
    assert "FUGU_API_KEY is not set" in capsys.readouterr().err


def test_all_cases_registered():
    assert {"screw_fastening", "solder_wetting", "dimensional_variation"} <= set(
        QUALITY_ISSUE_CASES
    )
