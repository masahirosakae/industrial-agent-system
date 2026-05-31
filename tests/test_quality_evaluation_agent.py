import json
from dataclasses import asdict

import pytest

from src.agents.countermeasure_planning_agent import (
    CountermeasurePlan,
    CountermeasurePlanningAgentOutput,
)
from src.agents.quality_evaluation_agent import (
    QUALITY_EVALUATION_OUTPUT_SCHEMA,
    QualityEvaluationAgent,
    QualityEvaluationAgentOutput,
    QualityEvaluationInput,
    QualityEvaluationResult,
)
from src.agents.root_cause_analysis_agent import (
    RootCauseAgentOutput,
    RootCauseAnalysis,
)
from src.llm.base import LLMProvider, LLMResponse
from src.llm.fugu_provider import FuguProvider
from src.llm.ollama_provider import OllamaProvider


# -------------------------
# Test providers
# -------------------------
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


# -------------------------
# Fixtures
# -------------------------
HYPOTHESIS_DRIVER = "DC driver transducer is drifting low"
HYPOTHESIS_OPERATOR = "Operator inconsistency at shift handover"


def make_root_cause_analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        problem_statement=(
            "Torque audit on screw_fastening shows 12% units below spec."
        ),
        facts=[
            {
                "fact": "post-assembly torque audit shows 12% units below spec",
                "source": "quality_issue_analysis.issue_summary",
            }
        ],
        assumptions=[],
        hypotheses=[
            {
                "hypothesis": HYPOTHESIS_DRIVER,
                "confidence": "high",
                "supporting_facts": ["torque audit 12% under spec"],
                "missing_evidence": ["recent calibration verification record"],
            },
            {
                "hypothesis": HYPOTHESIS_OPERATOR,
                "confidence": "low",
                "supporting_facts": [],
                "missing_evidence": ["shift-by-shift torque distribution"],
            },
        ],
        five_why=[
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
        fta_tree={
            "top_event": "Loose torque on assembled units",
            "branches": [
                {"category": "equipment", "causes": ["driver transducer drift"]},
            ],
        },
        missing_evidence=[
            {
                "evidence": "driver calibration log",
                "purpose": "confirm last verified torque value",
                "priority": "high",
            }
        ],
    )


def make_countermeasure_plan() -> CountermeasurePlan:
    return CountermeasurePlan(
        containment_actions=[
            {
                "action": "100% torque audit on last 4 hours of production",
                "target": "Aluminum housing M4 lot 2025-W22",
                "purpose": "Quarantine suspect units before shipment",
                "urgency": "high",
                "owner_candidate": "QC supervisor",
            }
        ],
        permanent_actions=[
            {
                "action": "Tighten driver calibration verification cadence to weekly",
                "related_hypothesis": HYPOTHESIS_DRIVER,
                "expected_effect": "Driver torque stays within +/- 5% of target",
                "implementation_difficulty": "medium",
            }
        ],
        verification_plan=[
            {
                "verification_item": "Driver torque vs reference transducer",
                "method": "Calibrated reference transducer sample test",
                "success_criteria": "All readings within +/- 5% of target torque",
                "required_data": [
                    "driver calibration log",
                    "recent calibration verification record",
                ],
                "related_hypothesis": HYPOTHESIS_DRIVER,
            },
            {
                "verification_item": "Shift-by-shift torque distribution comparison",
                "method": "SPC plot torque values grouped by shift",
                "success_criteria": "No statistically significant shift difference",
                "required_data": [
                    "torque trace log",
                    "shift-by-shift torque distribution",
                ],
                "related_hypothesis": HYPOTHESIS_OPERATOR,
            },
        ],
        risk_assessment=[
            {
                "risk": "Production downtime during driver recalibration",
                "impact": "low",
                "mitigation": "Schedule recalibration during off-shift window",
            }
        ],
        priority_recommendation=[
            {"priority": 1, "action": "Run 100% torque audit", "reason": "containment"},
            {"priority": 2, "action": "Recalibrate driver", "reason": "root cause"},
        ],
        review_reasons=[],
    )


def make_input(case_id: str = "qea-001") -> QualityEvaluationInput:
    return QualityEvaluationInput(
        case_id=case_id,
        root_cause_analysis=make_root_cause_analysis(),
        countermeasure_plan=make_countermeasure_plan(),
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )


def valid_data() -> dict:
    return {
        "evidence_sufficiency": {
            "score": 0.85,
            "assessment": (
                "Torque audit and calibration logs sufficiently support the "
                "primary hypothesis."
            ),
            "insufficient_items": [],
        },
        "hallucination_risk": {
            "risk_level": "low",
            "suspected_items": [],
        },
        "rca_cmp_consistency": {
            "score": 0.9,
            "assessment": (
                "Every CMP action references an RCA hypothesis; no unmapped "
                "causes detected."
            ),
            "inconsistencies": [],
        },
        "countermeasure_quality": {
            "score": 0.8,
            "assessment": (
                "Containment is immediate; permanent action addresses the "
                "high-confidence root cause."
            ),
            "weaknesses": [],
        },
        "verification_quality": {
            "score": 0.85,
            "assessment": (
                "Verification plan covers each hypothesis with measurable "
                "success criteria."
            ),
            "missing_verifications": [],
        },
        "approval_readiness": {
            "score": 0.85,
            "assessment": (
                "Analysis and plan are ready for engineering review and "
                "execution."
            ),
        },
        "overall_judgement": "accepted",
        "review_reasons": [],
        "recommended_next_steps": [
            "Execute the 100% torque audit on the affected lot",
            "Schedule driver recalibration during the next off-shift window",
        ],
    }


def response_text(data: dict) -> str:
    return json.dumps(data)


# =========================================================
# 1 + 2. Happy path + schema presence
# =========================================================
def test_schema_has_all_required_keys():
    expected = {
        "evidence_sufficiency",
        "hallucination_risk",
        "rca_cmp_consistency",
        "countermeasure_quality",
        "verification_quality",
        "approval_readiness",
        "overall_judgement",
        "review_reasons",
        "recommended_next_steps",
    }
    assert set(QUALITY_EVALUATION_OUTPUT_SCHEMA) == expected

    result = QualityEvaluationResult()
    for key in expected:
        assert hasattr(result, key)


def test_agent_calls_provider_and_parses_structured_result():
    provider = RecordingProvider(response_text(valid_data()))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert len(provider.prompts) == 1
    assert "root_cause_analysis" in provider.prompts[0]
    assert "countermeasure_plan" in provider.prompts[0]
    assert provider.system_prompts == [QualityEvaluationAgent.system_prompt]
    assert isinstance(output, QualityEvaluationAgentOutput)
    assert output.status == "success"
    assert output.needs_review is False
    assert output.review_reasons == []
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None
    assert isinstance(output.result, QualityEvaluationResult)
    assert output.result.overall_judgement == "accepted"
    assert output.result.approval_readiness["score"] == 0.85

    result_dict = asdict(output.result)
    for key in QUALITY_EVALUATION_OUTPUT_SCHEMA:
        assert key in result_dict


# =========================================================
# 3 + 4. Parse / schema fallback
# =========================================================
def test_invalid_json_response_falls_back_to_needs_review():
    provider = RecordingProvider("definitely not JSON")
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())  # must not raise

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"
    assert output.errors
    assert output.review_reasons and "parse_error" in output.review_reasons[0]
    # Conservative fallback: hallucination_risk forced to "high" and
    # overall_judgement forced to "needs_review".
    assert output.result.hallucination_risk["risk_level"] == "high"
    assert output.result.overall_judgement == "needs_review"
    assert output.result.approval_readiness["score"] == 0.0


def _without_key(data: dict, key: str) -> dict:
    return {k: v for k, v in data.items() if k != key}


@pytest.mark.parametrize(
    ("bad_data", "expected_error"),
    [
        ({}, "missing required keys"),
        (
            _without_key(valid_data(), "approval_readiness"),
            "missing required keys",
        ),
        (
            {**valid_data(), "hallucination_risk": {"risk_level": "low"}},
            "hallucination_risk.suspected_items must be a list",
        ),
        (
            {
                **valid_data(),
                "evidence_sufficiency": {
                    "score": 0.8,
                    "assessment": "",
                    "insufficient_items": [],
                },
            },
            "evidence_sufficiency.assessment must be a non-empty string",
        ),
        (
            {
                **valid_data(),
                "approval_readiness": {"score": 0.8, "assessment": ""},
            },
            "approval_readiness.assessment must be a non-empty string",
        ),
        (
            {**valid_data(), "recommended_next_steps": ["", "ok"]},
            "recommended_next_steps[0] must be a non-empty string",
        ),
    ],
)
def test_schema_mismatch_falls_back_to_needs_review(bad_data, expected_error):
    provider = RecordingProvider(response_text(bad_data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert expected_error in output.errors[0]


def test_empty_json_response_is_not_schema_valid():
    provider = RecordingProvider("{}")
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"


# =========================================================
# 5 / 6 / 7 / 8. Strict whole-response JSON parse
# =========================================================
def test_strict_parser_rejects_prefix_prose():
    provider = RecordingProvider("Here you go: " + response_text(valid_data()))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_suffix_prose():
    provider = RecordingProvider(response_text(valid_data()) + " Hope that helps!")
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_fenced_json():
    fenced = "```json\n" + response_text(valid_data()) + "\n```"
    provider = RecordingProvider(fenced)
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_accepts_pure_json_object():
    provider = RecordingProvider(response_text(valid_data()))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True


def test_strict_parser_allows_surrounding_whitespace_only():
    provider = RecordingProvider("  \n" + response_text(valid_data()) + "\n  ")
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True


# =========================================================
# 9 / 10 / 11 / 12. RCA + CMP input gate
# =========================================================
def _trusted_rca_output(**overrides) -> RootCauseAgentOutput:
    defaults = dict(
        case_id="rca-1",
        agent_name="root_cause_analysis_agent",
        status="success",
        result=make_root_cause_analysis(),
        confidence=0.7,
        errors=[],
        notes=[],
        needs_review=False,
        review_reasons=[],
        parse_success=True,
        schema_valid=True,
        error_type=None,
        raw_output="{}",
    )
    defaults.update(overrides)
    return RootCauseAgentOutput(**defaults)


def _trusted_cmp_output(**overrides) -> CountermeasurePlanningAgentOutput:
    defaults = dict(
        case_id="cmp-1",
        agent_name="countermeasure_planning_agent",
        status="success",
        result=make_countermeasure_plan(),
        confidence=0.7,
        errors=[],
        notes=[],
        needs_review=False,
        review_reasons=[],
        parse_success=True,
        schema_valid=True,
        error_type=None,
        raw_output="{}",
    )
    defaults.update(overrides)
    return CountermeasurePlanningAgentOutput(**defaults)


@pytest.mark.parametrize(
    ("override", "expected_fragment"),
    [
        ({"status": "needs_review"}, "status='needs_review'"),
        ({"needs_review": True}, "needs_review=True"),
        ({"parse_success": False}, "parse_success=False"),
        ({"schema_valid": False}, "schema_valid=False"),
    ],
)
def test_input_rejects_untrusted_rca_output(override, expected_fragment):
    prior_rca = _trusted_rca_output(**override)
    prior_cmp = _trusted_cmp_output()
    with pytest.raises(ValueError, match=expected_fragment):
        QualityEvaluationInput.from_previous_outputs(prior_rca, prior_cmp)


@pytest.mark.parametrize(
    ("override", "expected_fragment"),
    [
        ({"status": "needs_review"}, "status='needs_review'"),
        ({"needs_review": True}, "needs_review=True"),
        ({"parse_success": False}, "parse_success=False"),
        ({"schema_valid": False}, "schema_valid=False"),
    ],
)
def test_input_rejects_untrusted_cmp_output(override, expected_fragment):
    prior_rca = _trusted_rca_output()
    prior_cmp = _trusted_cmp_output(**override)
    with pytest.raises(ValueError, match=expected_fragment):
        QualityEvaluationInput.from_previous_outputs(prior_rca, prior_cmp)


def test_input_rejects_rca_result_is_none():
    prior_rca = RootCauseAgentOutput(
        case_id="rca-1",
        agent_name="root_cause_analysis_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["x"],
        notes=[],
        needs_review=True,
    )
    prior_cmp = _trusted_cmp_output()
    with pytest.raises(ValueError, match="RootCauseAgentOutput.result is required"):
        QualityEvaluationInput.from_previous_outputs(prior_rca, prior_cmp)


def test_input_rejects_cmp_result_is_none():
    prior_rca = _trusted_rca_output()
    prior_cmp = CountermeasurePlanningAgentOutput(
        case_id="cmp-1",
        agent_name="countermeasure_planning_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["x"],
        notes=[],
        needs_review=True,
    )
    with pytest.raises(
        ValueError,
        match="CountermeasurePlanningAgentOutput.result is required",
    ):
        QualityEvaluationInput.from_previous_outputs(prior_rca, prior_cmp)


def test_input_accepts_fully_trusted_outputs():
    prior_rca = _trusted_rca_output()
    prior_cmp = _trusted_cmp_output()

    eval_input = QualityEvaluationInput.from_previous_outputs(
        prior_rca, prior_cmp, process_name="p", product_or_part="q"
    )

    assert eval_input.case_id == "rca-1"
    assert eval_input.root_cause_analysis is prior_rca.result
    assert eval_input.countermeasure_plan is prior_cmp.result
    assert eval_input.process_name == "p"
    assert eval_input.product_or_part == "q"


# =========================================================
# 13. score range validation
# =========================================================
@pytest.mark.parametrize("bad_score", [-0.1, 1.5, -1.0])
def test_score_out_of_range_is_schema_validation_failure(bad_score):
    data = valid_data()
    data["evidence_sufficiency"]["score"] = bad_score
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert (
        "evidence_sufficiency.score must be between 0.0 and 1.0"
        in output.errors[0]
    )


# 14. bool scores
def test_score_as_bool_is_schema_validation_failure():
    data = valid_data()
    data["approval_readiness"]["score"] = True  # bool sneaks past int subclass
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert "approval_readiness.score must be a number, got bool" in output.errors[0]


# =========================================================
# 15 + 16. enum validation
# =========================================================
def test_invalid_risk_level_is_rejected():
    data = valid_data()
    data["hallucination_risk"]["risk_level"] = "extreme"
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert "hallucination_risk.risk_level must be one of" in output.errors[0]


def test_invalid_overall_judgement_is_rejected():
    data = valid_data()
    data["overall_judgement"] = "approved"
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert "overall_judgement must be one of" in output.errors[0]


# =========================================================
# 17. hallucination_risk="high" forces needs_review override
# =========================================================
def test_high_hallucination_risk_overrides_accepted_to_needs_review():
    data = valid_data()
    data["hallucination_risk"]["risk_level"] = "high"
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type == "policy_review_required"
    assert output.result.overall_judgement == "needs_review"
    assert any(
        "hallucination_risk.risk_level is high" in reason
        for reason in output.review_reasons
    )


# =========================================================
# 18. any score < 0.5 forces needs_review override
# =========================================================
@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("evidence_sufficiency", "score"),
        ("rca_cmp_consistency", "score"),
        ("countermeasure_quality", "score"),
        ("verification_quality", "score"),
        ("approval_readiness", "score"),
    ],
)
def test_low_score_overrides_accepted_to_needs_review(section, field):
    data = valid_data()
    data[section][field] = 0.3
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "policy_review_required"
    assert output.result.overall_judgement == "needs_review"
    assert any(
        f"{section}.{field} (0.3) is below threshold 0.5" in reason
        for reason in output.review_reasons
    )


# =========================================================
# 19. non-empty review_reasons forces needs_review override
# =========================================================
def test_non_empty_review_reasons_overrides_accepted_to_needs_review():
    data = valid_data()
    data["review_reasons"] = ["LLM uncertainty about driver supplier behavior"]
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "policy_review_required"
    assert output.result.overall_judgement == "needs_review"
    assert "LLM uncertainty about driver supplier behavior" in output.review_reasons
    assert any(
        "review_reasons is non-empty" in reason for reason in output.review_reasons
    )


# =========================================================
# 20. non-empty suspected_items forces needs_review override
# =========================================================
def test_non_empty_suspected_items_overrides_accepted_to_needs_review():
    data = valid_data()
    data["hallucination_risk"]["suspected_items"] = [
        {
            "item": "Suggested 'laser micrometer' check",
            "reason": "No laser micrometer is mentioned in the RCA or CMP",
        }
    ]
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "policy_review_required"
    assert any(
        "hallucination_risk.suspected_items is non-empty" in reason
        for reason in output.review_reasons
    )


# =========================================================
# 21. other non-empty negative-list fields force needs_review override
# =========================================================
@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("evidence_sufficiency", "insufficient_items"),
        ("rca_cmp_consistency", "inconsistencies"),
        ("countermeasure_quality", "weaknesses"),
        ("verification_quality", "missing_verifications"),
    ],
)
def test_negative_list_fields_force_needs_review(section, field):
    data = valid_data()
    data[section][field] = ["something is off"]
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "policy_review_required"
    assert output.result.overall_judgement == "needs_review"
    assert any(
        f"{section}.{field} is non-empty" in reason
        for reason in output.review_reasons
    )


# When LLM directly returns needs_review (no overrides), agent still flips
# status to needs_review.
def test_model_emits_needs_review_judgement_propagates():
    data = valid_data()
    data["overall_judgement"] = "needs_review"
    data["review_reasons"] = ["Engineer review requested by analyst convention"]
    provider = RecordingProvider(response_text(data))
    agent = QualityEvaluationAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type == "policy_review_required"
    assert output.result.overall_judgement == "needs_review"
    assert "Engineer review requested by analyst convention" in output.review_reasons


# =========================================================
# 22. Provider-agnostic
# =========================================================
def test_provider_exception_returns_needs_review_without_crash():
    agent = QualityEvaluationAgent(provider=BoomProvider())

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "provider_error"
    assert output.errors == ["provider blew up"]
    assert output.result.hallucination_risk["risk_level"] == "high"
    assert output.result.overall_judgement == "needs_review"


def test_agent_defaults_to_ollama_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    agent = QualityEvaluationAgent()
    assert isinstance(agent.provider, OllamaProvider)


def test_agent_accepts_fugu_provider_name(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    agent = QualityEvaluationAgent(provider_name="fugu")
    assert isinstance(agent.provider, FuguProvider)


def test_agent_module_does_not_import_fugu_or_ollama_directly():
    import src.agents.quality_evaluation_agent.quality_evaluation_agent as module

    source = module.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "FuguProvider" not in text
    assert "OllamaProvider" not in text
    assert "fugu_provider" not in text
    assert "ollama_provider" not in text
