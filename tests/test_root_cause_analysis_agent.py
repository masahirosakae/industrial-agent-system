import json
from dataclasses import asdict

import pytest

from src.agents.quality_issue_analysis_agent import (
    QualityIssueAgentOutput,
    QualityIssueAnalysis,
)
from src.agents.root_cause_analysis_agent import (
    ROOT_CAUSE_OUTPUT_SCHEMA,
    RootCauseAgentOutput,
    RootCauseAnalysis,
    RootCauseAnalysisAgent,
    RootCauseAnalysisInput,
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


def make_quality_issue_analysis() -> QualityIssueAnalysis:
    return QualityIssueAnalysis(
        issue_summary="Loose torque on aluminum housing M4 joint",
        suspected_causes=[
            {
                "cause": "driver calibration drift",
                "evidence": "torque audit 12% under spec",
                "likelihood": "high",
            }
        ],
        containment_actions=["100% torque audit on last 4 hours of production"],
        investigation_plan=[
            {
                "step": "verify driver torque against reference transducer",
                "owner": "maintenance",
                "expected_signal": "deviation within tolerance",
            }
        ],
        additional_data_needed=["driver calibration log"],
        risk_if_unresolved="field returns with loose joints",
        verification_points=["torque Cpk recovers >= 1.33"],
    )


def make_input(case_id: str = "rca-001") -> RootCauseAnalysisInput:
    return RootCauseAnalysisInput(
        case_id=case_id,
        quality_issue_analysis=make_quality_issue_analysis(),
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )


def valid_data() -> dict:
    return {
        "problem_statement": (
            "Torque audit on screw_fastening shows 12% units below spec."
        ),
        "facts": [
            {
                "fact": "post-assembly torque audit shows 12% units below spec",
                "source": "quality_issue_analysis.issue_summary",
            }
        ],
        "assumptions": [
            {
                "assumption": "Driver calibration drift is the dominant mode",
                "reason": "consistent with high-likelihood suspected cause",
            }
        ],
        "hypotheses": [
            {
                "hypothesis": "DC driver torque transducer is drifting low",
                "confidence": "high",
                "supporting_facts": ["torque audit 12% under spec"],
                "missing_evidence": ["recent calibration verification record"],
            }
        ],
        "five_why": [
            {
                "level": 1,
                "why": "Why are units below torque spec?",
                "answer": "Driver delivers torque below target",
                "evidence_status": "confirmed",
            },
            {
                "level": 2,
                "why": "Why is the driver below target?",
                "answer": "Calibration has drifted",
                "evidence_status": "assumed",
            },
            {
                "level": 3,
                "why": "Why has calibration drifted?",
                "answer": "Verification interval may have been exceeded",
                "evidence_status": "missing",
            },
        ],
        "fta_tree": {
            "top_event": "Loose torque on assembled units",
            "branches": [
                {
                    "category": "equipment",
                    "causes": ["driver transducer drift"],
                },
                {
                    "category": "process",
                    "causes": ["missed calibration verification"],
                },
            ],
        },
        "missing_evidence": [
            {
                "evidence": "driver calibration log",
                "purpose": "confirm last verified torque value",
                "priority": "high",
            }
        ],
    }


def response_text(data: dict) -> str:
    return json.dumps(data)


# -------------------------
# 1. schema dataclass presence
# -------------------------
def test_schema_has_all_required_keys():
    expected = {
        "problem_statement",
        "facts",
        "assumptions",
        "hypotheses",
        "five_why",
        "fta_tree",
        "missing_evidence",
    }
    assert set(ROOT_CAUSE_OUTPUT_SCHEMA) == expected

    analysis = RootCauseAnalysis(problem_statement="x")
    for key in expected:
        assert hasattr(analysis, key)


# -------------------------
# 2. provider invocation + structured success result
# -------------------------
def test_agent_calls_provider_and_parses_structured_result():
    provider = RecordingProvider(response_text(valid_data()))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert len(provider.prompts) == 1
    assert "quality_issue_analysis" in provider.prompts[0]
    assert provider.system_prompts == [RootCauseAnalysisAgent.system_prompt]
    assert output.status == "success"
    assert output.needs_review is False
    assert output.review_reasons == []
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None
    assert isinstance(output, RootCauseAgentOutput)
    assert isinstance(output.result, RootCauseAnalysis)

    result_dict = asdict(output.result)
    for key in ROOT_CAUSE_OUTPUT_SCHEMA:
        assert key in result_dict
    # Required-field presence within the dataclass values
    assert result_dict["problem_statement"]
    assert result_dict["hypotheses"][0]["confidence"] == "high"
    assert result_dict["five_why"][0]["evidence_status"] == "confirmed"
    assert result_dict["fta_tree"]["top_event"]
    assert result_dict["fta_tree"]["branches"][0]["category"] == "equipment"


# -------------------------
# 3. invalid JSON -> needs_review fallback
# -------------------------
def test_invalid_json_response_falls_back_to_needs_review():
    provider = RecordingProvider("definitely not JSON")
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"
    assert output.errors
    assert output.review_reasons and "parse_error" in output.review_reasons[0]
    assert output.raw_output == "definitely not JSON"
    # Fallback shape matches the requested empty layout
    assert output.result is not None
    fallback = asdict(output.result)
    assert fallback == {
        "problem_statement": "",
        "facts": [],
        "assumptions": [],
        "hypotheses": [],
        "five_why": [],
        "fta_tree": {"top_event": "", "branches": []},
        "missing_evidence": [],
    }


# -------------------------
# 4. schema-mismatched JSON -> needs_review fallback
# -------------------------
@pytest.mark.parametrize(
    ("bad_data", "expected_error"),
    [
        ({}, "missing required keys"),
        (
            {key: value for key, value in valid_data().items() if key != "problem_statement"},
            "missing required keys",
        ),
        ({**valid_data(), "hypotheses": []}, "hypotheses must not be empty"),
        (
            {
                **valid_data(),
                "hypotheses": [
                    {
                        "hypothesis": "h",
                        "confidence": "certain",
                        "supporting_facts": ["a"],
                        "missing_evidence": [],
                    }
                ],
            },
            "confidence",
        ),
        (
            {
                **valid_data(),
                "five_why": [
                    {
                        "level": 1,
                        "why": "w",
                        "answer": "a",
                        "evidence_status": "unknown",
                    }
                ],
            },
            "evidence_status",
        ),
        (
            {
                **valid_data(),
                "fta_tree": {"top_event": "", "branches": []},
            },
            "fta_tree",
        ),
        (
            {
                **valid_data(),
                "missing_evidence": [
                    {"evidence": "e", "purpose": "p", "priority": "urgent"}
                ],
            },
            "priority",
        ),
    ],
)
def test_schema_mismatch_falls_back_to_needs_review(bad_data, expected_error):
    provider = RecordingProvider(response_text(bad_data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert expected_error in output.errors[0]


def test_empty_json_response_is_not_schema_valid():
    provider = RecordingProvider("{}")
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"


# -------------------------
# 5. Provider-agnostic (no dependency on FuguProvider / OllamaProvider internals)
# -------------------------
def test_provider_exception_returns_needs_review_without_crash():
    agent = RootCauseAnalysisAgent(provider=BoomProvider())

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.error_type == "provider_error"
    assert output.errors == ["provider blew up"]


def test_agent_defaults_to_ollama_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    agent = RootCauseAnalysisAgent()
    assert isinstance(agent.provider, OllamaProvider)


def test_agent_accepts_fugu_provider_name(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    agent = RootCauseAnalysisAgent(provider_name="fugu")
    assert isinstance(agent.provider, FuguProvider)


def test_agent_module_does_not_import_fugu_or_ollama_directly():
    import src.agents.root_cause_analysis_agent.root_cause_analysis_agent as module

    source = module.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "FuguProvider" not in text
    assert "OllamaProvider" not in text
    assert "fugu_provider" not in text
    assert "ollama_provider" not in text


# -------------------------
# Input builder consumes the prior agent's output
# -------------------------
def test_input_from_quality_issue_output_round_trip():
    prior = QualityIssueAgentOutput(
        case_id="qi-1",
        agent_name="quality_issue_analysis_agent",
        status="success",
        result=make_quality_issue_analysis(),
        confidence=0.7,
        errors=[],
        notes=[],
        needs_review=False,
        parse_success=True,
        schema_valid=True,
        error_type=None,
        raw_output="{}",
    )

    rca_input = RootCauseAnalysisInput.from_quality_issue_output(
        prior,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )

    assert rca_input.case_id == "qi-1"
    assert rca_input.quality_issue_analysis.issue_summary.startswith("Loose torque")


def test_input_from_quality_issue_output_rejects_missing_result():
    prior = QualityIssueAgentOutput(
        case_id="qi-1",
        agent_name="quality_issue_analysis_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["x"],
        notes=[],
        needs_review=True,
    )
    with pytest.raises(ValueError, match="result is required"):
        RootCauseAnalysisInput.from_quality_issue_output(prior)


# -------------------------
# QIA -> RCA fallback gate: reject untrusted prior analyses even when result is non-None
# -------------------------
def _trusted_qia_output(**overrides) -> QualityIssueAgentOutput:
    defaults = dict(
        case_id="qi-1",
        agent_name="quality_issue_analysis_agent",
        status="success",
        result=make_quality_issue_analysis(),
        confidence=0.7,
        errors=[],
        notes=[],
        needs_review=False,
        parse_success=True,
        schema_valid=True,
        error_type=None,
        raw_output="{}",
    )
    defaults.update(overrides)
    return QualityIssueAgentOutput(**defaults)


def test_input_rejects_qia_needs_review_status_even_with_result():
    prior = _trusted_qia_output(status="needs_review")
    with pytest.raises(ValueError, match=r"status='needs_review'"):
        RootCauseAnalysisInput.from_quality_issue_output(prior)


def test_input_rejects_qia_with_needs_review_flag():
    prior = _trusted_qia_output(needs_review=True)
    with pytest.raises(ValueError, match="needs_review=True"):
        RootCauseAnalysisInput.from_quality_issue_output(prior)


def test_input_rejects_qia_with_parse_failure():
    prior = _trusted_qia_output(parse_success=False)
    with pytest.raises(ValueError, match="parse_success=False"):
        RootCauseAnalysisInput.from_quality_issue_output(prior)


def test_input_rejects_qia_with_schema_invalid():
    prior = _trusted_qia_output(schema_valid=False)
    with pytest.raises(ValueError, match="schema_valid=False"):
        RootCauseAnalysisInput.from_quality_issue_output(prior)


def test_input_accepts_fully_trusted_qia_output():
    prior = _trusted_qia_output()
    rca_input = RootCauseAnalysisInput.from_quality_issue_output(
        prior,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )
    assert rca_input.case_id == "qi-1"
    assert rca_input.process_name == "screw_fastening"
    assert rca_input.product_or_part == "aluminum housing M4"
    assert rca_input.quality_issue_analysis is prior.result


# -------------------------
# Strict whole-response JSON parse
# -------------------------
def test_strict_parser_rejects_prefix_prose():
    provider = RecordingProvider("Here you go: " + response_text(valid_data()))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_suffix_prose():
    provider = RecordingProvider(response_text(valid_data()) + " Hope that helps!")
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_fenced_json():
    fenced = "```json\n" + response_text(valid_data()) + "\n```"
    provider = RecordingProvider(fenced)
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"


def test_strict_parser_accepts_pure_json_object():
    provider = RecordingProvider(response_text(valid_data()))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None


def test_strict_parser_allows_surrounding_whitespace_only():
    # ``json.loads`` itself treats surrounding whitespace as JSON-valid.
    provider = RecordingProvider("  \n" + response_text(valid_data()) + "\n  ")
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True


# -------------------------
# five_why count policy: 3..5 entries
# -------------------------
def _make_five_why(count: int) -> list[dict]:
    statuses = ["confirmed", "assumed", "missing"]
    return [
        {
            "level": index + 1,
            "why": f"why question {index + 1}",
            "answer": f"answer {index + 1}",
            "evidence_status": statuses[index % len(statuses)],
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("count", [1, 2, 6])
def test_five_why_invalid_counts_fall_back_to_needs_review(count):
    data = {**valid_data(), "five_why": _make_five_why(count)}
    provider = RecordingProvider(response_text(data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert "five_why must have between 3 and 5 entries" in output.errors[0]


@pytest.mark.parametrize("count", [3, 5])
def test_five_why_valid_counts_succeed(count):
    data = {**valid_data(), "five_why": _make_five_why(count)}
    provider = RecordingProvider(response_text(data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True



# -------------------------
# five_why.level must form an exact 1..N sequence
# -------------------------
def _five_why_with_levels(levels: list) -> list[dict]:
    statuses = ["confirmed", "assumed", "missing"]
    return [
        {
            "level": level,
            "why": f"why question {index + 1}",
            "answer": f"answer {index + 1}",
            "evidence_status": statuses[index % len(statuses)],
        }
        for index, level in enumerate(levels)
    ]


@pytest.mark.parametrize(
    "levels",
    [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ],
)
def test_five_why_levels_exactly_one_to_n_succeed(levels):
    data = {**valid_data(), "five_why": _five_why_with_levels(levels)}
    provider = RecordingProvider(response_text(data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None


@pytest.mark.parametrize(
    ("levels", "label"),
    [
        ([1, 1, 3], "duplicate"),
        ([1, 3, 4], "gap"),
        ([3, 2, 1], "reverse"),
        ([0, 1, 2], "zero_start"),
        ([1, 2, 99], "overflow"),
    ],
    ids=lambda value: value if isinstance(value, str) else "-".join(str(v) for v in value),
)
def test_five_why_levels_must_form_one_to_n_sequence(levels, label):
    data = {**valid_data(), "five_why": _five_why_with_levels(levels)}
    provider = RecordingProvider(response_text(data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert "five_why levels must be a 1..N sequence" in output.errors[0]


def test_five_why_bool_level_is_rejected_as_non_positive_integer():
    # ``bool`` is a subclass of ``int`` in Python, so guard explicitly.
    bad_item = {
        "level": True,
        "why": "w",
        "answer": "a",
        "evidence_status": "confirmed",
    }
    other = _five_why_with_levels([2, 3])
    data = {**valid_data(), "five_why": [bad_item, *other]}
    provider = RecordingProvider(response_text(data))
    agent = RootCauseAnalysisAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert "five_why[0].level must be a positive integer" in output.errors[0]
