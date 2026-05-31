import json
from dataclasses import asdict

import pytest

from src.agents.quality_issue_analysis_agent import (
    QUALITY_ISSUE_OUTPUT_SCHEMA,
    QualityIssueAnalysis,
    QualityIssueAnalysisAgent,
    QualityIssueInput,
)
from src.llm.base import LLMProvider, LLMResponse
from src.llm.fugu_provider import FuguProvider
from src.llm.ollama_provider import OllamaProvider


class RecordingProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return LLMResponse(text=self.text, model="test-model", provider="test")


class BoomProvider(LLMProvider):
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        raise RuntimeError("provider blew up")


def make_input(case_id: str = "qi-001") -> QualityIssueInput:
    return QualityIssueInput(
        case_id=case_id,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
        defect_mode="loose torque",
        observed_symptoms=["torque audit shows 12% units below spec"],
        process_conditions={"target_torque_Nm": 1.2},
        known_constraints=["screw spec frozen"],
        available_data=["torque trace log"],
    )


def valid_data() -> dict:
    return {
        "issue_summary": "Loose torque traced to driver calibration drift.",
        "suspected_causes": [
            {
                "cause": "driver calibration drift",
                "evidence": "torque audit 12% under spec",
                "likelihood": "high",
            }
        ],
        "containment_actions": ["100% torque audit on last 4 hours of production"],
        "investigation_plan": [
            {
                "step": "verify driver torque against reference transducer",
                "owner": "maintenance",
                "expected_signal": "deviation within tolerance",
            }
        ],
        "additional_data_needed": ["driver calibration log"],
        "risk_if_unresolved": "field returns with loose joints",
        "verification_points": ["torque Cpk recovers >= 1.33"],
    }


def response_text(data: dict) -> str:
    return json.dumps(data)


def test_schema_has_all_required_keys():
    expected = {
        "issue_summary",
        "suspected_causes",
        "containment_actions",
        "investigation_plan",
        "additional_data_needed",
        "risk_if_unresolved",
        "verification_points",
    }
    assert set(QUALITY_ISSUE_OUTPUT_SCHEMA) == expected

    analysis = QualityIssueAnalysis(issue_summary="x")
    for key in expected:
        assert hasattr(analysis, key)


def test_agent_calls_provider_and_parses_structured_result():
    provider = RecordingProvider(response_text(valid_data()))
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert len(provider.prompts) == 1
    assert "screw_fastening" in provider.prompts[0]
    assert provider.system_prompts == [QualityIssueAnalysisAgent.system_prompt]
    assert output.status == "success"
    assert output.needs_review is False
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None
    assert output.result is not None
    assert isinstance(output.result, QualityIssueAnalysis)
    assert output.result.suspected_causes[0]["likelihood"] == "high"
    result_dict = asdict(output.result)
    for key in QUALITY_ISSUE_OUTPUT_SCHEMA:
        assert key in result_dict


def test_agent_falls_back_on_invalid_json_as_needs_review():
    provider = RecordingProvider("this is not JSON at all")
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"
    assert output.errors
    assert output.raw_output == "this is not JSON at all"
    assert output.result is not None
    assert "Do not use this fallback" in output.result.containment_actions[0]


@pytest.mark.parametrize(
    ("bad_data", "expected_error"),
    [
        ({}, "missing required keys"),
        (
            {key: value for key, value in valid_data().items() if key != "issue_summary"},
            "missing required keys",
        ),
        (
            {**valid_data(), "suspected_causes": {"cause": "not-list"}},
            "suspected_causes must be a list",
        ),
        (
            {
                **valid_data(),
                "suspected_causes": [
                    {"cause": "c", "evidence": "e", "likelihood": "certain"}
                ],
            },
            "likelihood",
        ),
        ({**valid_data(), "suspected_causes": []}, "suspected_causes must not be empty"),
        ({**valid_data(), "investigation_plan": []}, "investigation_plan must not be empty"),
    ],
)
def test_malformed_structured_output_is_not_schema_valid(bad_data, expected_error):
    provider = RecordingProvider(response_text(bad_data))
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert expected_error in output.errors[0]


def test_empty_json_response_is_not_schema_valid():
    provider = RecordingProvider("{}")
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"


def test_agent_handles_provider_exception_as_needs_review_not_success():
    agent = QualityIssueAnalysisAgent(provider=BoomProvider())

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "provider_error"
    assert output.errors == ["provider blew up"]
    assert output.result is not None
    result_dict = asdict(output.result)
    for key in QUALITY_ISSUE_OUTPUT_SCHEMA:
        assert key in result_dict


@pytest.mark.parametrize(
    "wrapped_response",
    [
        lambda text: f"Here is the JSON: {text}",
        lambda text: f"{text}\nAnalysis complete.",
    ],
)
def test_agent_rejects_json_with_surrounding_prose(wrapped_response):
    provider = RecordingProvider(wrapped_response(response_text(valid_data())))
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"


def test_agent_rejects_code_fenced_json():
    fenced = "```json\n" + response_text(valid_data()) + "\n```"
    provider = RecordingProvider(fenced)
    agent = QualityIssueAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "non_json_response"
    assert output.result is not None


def test_agent_defaults_to_ollama_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    agent = QualityIssueAnalysisAgent()
    assert isinstance(agent.provider, OllamaProvider)


def test_agent_accepts_fugu_provider_name(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    agent = QualityIssueAnalysisAgent(provider_name="fugu")

    assert isinstance(agent.provider, FuguProvider)


@pytest.mark.parametrize(
    "case_name",
    ["screw_fastening", "solder_wetting", "dimensional_variation"],
)
def test_runner_cases_are_complete(case_name):
    from scripts.run_quality_issue_case import QUALITY_ISSUE_CASES

    case = QUALITY_ISSUE_CASES[case_name]
    assert case.process_name
    assert case.product_or_part
    assert case.defect_mode
    assert case.observed_symptoms
    assert case.process_conditions
    assert case.available_data
