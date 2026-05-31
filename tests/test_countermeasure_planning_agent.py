import json
from dataclasses import asdict

import pytest

from src.agents.countermeasure_planning_agent import (
    COUNTERMEASURE_PLANNING_OUTPUT_SCHEMA,
    CountermeasurePlan,
    CountermeasurePlanningAgent,
    CountermeasurePlanningAgentOutput,
    CountermeasurePlanningInput,
)
from src.agents.root_cause_analysis_agent import (
    RootCauseAgentOutput,
    RootCauseAnalysis,
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
        assumptions=[
            {
                "assumption": "Driver calibration drift is the dominant mode",
                "reason": "consistent with high-likelihood suspected cause",
            }
        ],
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
                {"category": "process", "causes": ["missed calibration verification"]},
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


def make_input(case_id: str = "cmp-001") -> CountermeasurePlanningInput:
    return CountermeasurePlanningInput(
        case_id=case_id,
        root_cause_analysis=make_root_cause_analysis(),
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )


def valid_data() -> dict:
    return {
        "containment_actions": [
            {
                "action": "100% torque audit on last 4 hours of production",
                "target": "Aluminum housing M4 lot 2025-W22",
                "purpose": "Quarantine suspect units before shipment",
                "urgency": "high",
                "owner_candidate": "QC supervisor",
            }
        ],
        "permanent_actions": [
            {
                "action": "Tighten driver calibration verification cadence to weekly",
                "related_hypothesis": HYPOTHESIS_DRIVER,
                "expected_effect": "Driver torque stays within +/- 5% of target",
                "implementation_difficulty": "medium",
            }
        ],
        "verification_plan": [
            {
                "verification_item": "Driver torque vs reference transducer",
                "method": "Calibrated reference transducer sample test",
                "success_criteria": "All readings within +/- 5% of target torque",
                "required_data": [
                    "driver calibration log",
                    "recent calibration verification record",
                    "reference transducer log",
                ],
                "related_hypothesis": HYPOTHESIS_DRIVER,
            },
            {
                "verification_item": "Shift-by-shift torque distribution comparison",
                "method": "SPC plot torque values grouped by shift",
                "success_criteria": "No statistically significant shift difference (p > 0.05)",
                "required_data": [
                    "torque trace log",
                    "shift roster",
                    "shift-by-shift torque distribution",
                ],
                "related_hypothesis": HYPOTHESIS_OPERATOR,
            },
        ],
        "risk_assessment": [
            {
                "risk": "Production downtime during driver recalibration",
                "impact": "low",
                "mitigation": "Schedule recalibration during off-shift window",
            }
        ],
        "priority_recommendation": [
            {
                "priority": 1,
                "action": "Run 100% torque audit",
                "reason": "Immediate containment of suspect units",
            },
            {
                "priority": 2,
                "action": "Recalibrate driver and tighten verification cadence",
                "reason": "Address the high-confidence root cause hypothesis",
            },
            {
                "priority": 3,
                "action": "Compare shift-by-shift torque distributions",
                "reason": "Confirm or rule out the low-confidence operator hypothesis",
            },
        ],
        "review_reasons": [],
    }


def response_text(data: dict) -> str:
    return json.dumps(data)


# =========================
# 1 + 2. Schema / required-field presence
# =========================
def test_schema_has_all_required_keys():
    expected = {
        "containment_actions",
        "permanent_actions",
        "verification_plan",
        "risk_assessment",
        "priority_recommendation",
        "review_reasons",
    }
    assert set(COUNTERMEASURE_PLANNING_OUTPUT_SCHEMA) == expected

    plan = CountermeasurePlan()
    for key in expected:
        assert hasattr(plan, key)


# =========================
# 1. Happy path: success + structured result
# =========================
def test_agent_calls_provider_and_parses_structured_result():
    provider = RecordingProvider(response_text(valid_data()))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert len(provider.prompts) == 1
    assert "root_cause_analysis" in provider.prompts[0]
    assert provider.system_prompts == [CountermeasurePlanningAgent.system_prompt]
    assert isinstance(output, CountermeasurePlanningAgentOutput)
    assert output.status == "success"
    assert output.needs_review is False
    assert output.review_reasons == []
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type is None
    assert isinstance(output.result, CountermeasurePlan)

    result_dict = asdict(output.result)
    for key in COUNTERMEASURE_PLANNING_OUTPUT_SCHEMA:
        assert key in result_dict
    assert result_dict["review_reasons"] == []
    assert result_dict["containment_actions"][0]["urgency"] == "high"
    assert (
        result_dict["permanent_actions"][0]["implementation_difficulty"] == "medium"
    )
    assert result_dict["risk_assessment"][0]["impact"] == "low"
    assert [
        item["priority"] for item in result_dict["priority_recommendation"]
    ] == [1, 2, 3]


# =========================
# 3. Invalid JSON -> needs_review fallback (no crash)
# =========================
def test_invalid_json_response_falls_back_to_needs_review():
    provider = RecordingProvider("definitely not JSON")
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())  # must not raise

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is False
    assert output.schema_valid is False
    assert output.error_type == "parse_error"
    assert output.errors
    assert output.review_reasons and "parse_error" in output.review_reasons[0]
    # Conservative empty envelope per the spec.
    assert asdict(output.result) == {
        "containment_actions": [],
        "permanent_actions": [],
        "verification_plan": [],
        "risk_assessment": [],
        "priority_recommendation": [],
        "review_reasons": [],
    }


# =========================
# 4. Schema mismatch -> needs_review fallback (multiple cases)
# =========================
def _without_key(data: dict, key: str) -> dict:
    return {k: v for k, v in data.items() if k != key}


@pytest.mark.parametrize(
    ("bad_data", "expected_error"),
    [
        ({}, "missing required keys"),
        (_without_key(valid_data(), "containment_actions"), "missing required keys"),
        (
            {**valid_data(), "containment_actions": []},
            "containment_actions must not be empty",
        ),
        (
            {**valid_data(), "verification_plan": []},
            "verification_plan must not be empty",
        ),
        (
            {**valid_data(), "risk_assessment": []},
            "risk_assessment must not be empty",
        ),
        (
            {**valid_data(), "priority_recommendation": []},
            "priority_recommendation must not be empty",
        ),
        (
            {
                **valid_data(),
                "verification_plan": [
                    {
                        "verification_item": "x",
                        "method": "m",
                        "success_criteria": "c",
                        "required_data": [],
                        "related_hypothesis": HYPOTHESIS_DRIVER,
                    }
                ],
            },
            "required_data must be a non-empty list",
        ),
        (
            {
                **valid_data(),
                "permanent_actions": [
                    {
                        "action": "do something",
                        "related_hypothesis": "totally invented hypothesis",
                        "expected_effect": "stuff",
                        "implementation_difficulty": "medium",
                    }
                ],
            },
            "related_hypothesis must match one of the input RCA hypotheses",
        ),
    ],
)
def test_schema_mismatch_falls_back_to_needs_review(bad_data, expected_error):
    provider = RecordingProvider(response_text(bad_data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert expected_error in output.errors[0]


def test_empty_json_response_is_not_schema_valid():
    provider = RecordingProvider("{}")
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"


# =========================
# 5/6/7/8. Strict whole-response JSON parse policy (mirrors RCA)
# =========================
def test_strict_parser_rejects_prefix_prose():
    provider = RecordingProvider("Here you go: " + response_text(valid_data()))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_suffix_prose():
    provider = RecordingProvider(response_text(valid_data()) + " Hope that helps!")
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_rejects_fenced_json():
    fenced = "```json\n" + response_text(valid_data()) + "\n```"
    provider = RecordingProvider(fenced)
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is False
    assert output.error_type == "parse_error"


def test_strict_parser_accepts_pure_json_object():
    provider = RecordingProvider(response_text(valid_data()))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True


def test_strict_parser_allows_surrounding_whitespace_only():
    provider = RecordingProvider("  \n" + response_text(valid_data()) + "\n  ")
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.parse_success is True
    assert output.schema_valid is True


# =========================
# 9. RCA fallback (untrusted prior analysis) is rejected at the input gate
# =========================
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


def test_input_rejects_rca_needs_review_status_even_with_result():
    prior = _trusted_rca_output(status="needs_review")
    with pytest.raises(ValueError, match=r"status='needs_review'"):
        CountermeasurePlanningInput.from_root_cause_output(prior)


def test_input_rejects_rca_with_needs_review_flag():
    prior = _trusted_rca_output(needs_review=True)
    with pytest.raises(ValueError, match="needs_review=True"):
        CountermeasurePlanningInput.from_root_cause_output(prior)


def test_input_rejects_rca_with_parse_failure():
    prior = _trusted_rca_output(parse_success=False)
    with pytest.raises(ValueError, match="parse_success=False"):
        CountermeasurePlanningInput.from_root_cause_output(prior)


def test_input_rejects_rca_with_schema_invalid():
    prior = _trusted_rca_output(schema_valid=False)
    with pytest.raises(ValueError, match="schema_valid=False"):
        CountermeasurePlanningInput.from_root_cause_output(prior)


def test_input_accepts_fully_trusted_rca_output():
    prior = _trusted_rca_output()
    cmp_input = CountermeasurePlanningInput.from_root_cause_output(
        prior,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )
    assert cmp_input.case_id == "rca-1"
    assert cmp_input.root_cause_analysis is prior.result
    assert cmp_input.process_name == "screw_fastening"


# =========================
# 10. result is None must be rejected (separate, explicit guard)
# =========================
def test_input_rejects_result_is_none():
    prior = RootCauseAgentOutput(
        case_id="rca-1",
        agent_name="root_cause_analysis_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["x"],
        notes=[],
        needs_review=True,
    )
    with pytest.raises(ValueError, match="result is required"):
        CountermeasurePlanningInput.from_root_cause_output(prior)


# =========================
# 11. priority_recommendation.priority must form exact 1..N
# =========================
def _priorities(priorities: list) -> list[dict]:
    return [
        {
            "priority": priority,
            "action": f"action {index + 1}",
            "reason": f"reason {index + 1}",
        }
        for index, priority in enumerate(priorities)
    ]


@pytest.mark.parametrize(
    ("priorities", "label"),
    [
        ([1, 1, 3], "duplicate"),
        ([1, 3, 4], "gap"),
        ([3, 2, 1], "reverse"),
        ([0, 1, 2], "zero_start"),
        ([1, 2, 99], "overflow"),
    ],
)
def test_priority_recommendation_invalid_sequences_rejected(priorities, label):
    data = {**valid_data(), "priority_recommendation": _priorities(priorities)}
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert output.error_type == "schema_validation_error"
    assert (
        "priority_recommendation priorities must be a 1..N sequence"
        in output.errors[0]
    )


def test_priority_recommendation_rejects_bool_priority():
    bad = {
        "priority": True,
        "action": "a",
        "reason": "r",
    }
    others = _priorities([2, 3])
    data = {**valid_data(), "priority_recommendation": [bad, *others]}
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.parse_success is True
    assert output.schema_valid is False
    assert (
        "priority_recommendation[0].priority must be a positive integer"
        in output.errors[0]
    )


@pytest.mark.parametrize("priorities", [[1, 2, 3], [1, 2, 3, 4, 5]])
def test_priority_recommendation_one_to_n_succeeds(priorities):
    data = {**valid_data(), "priority_recommendation": _priorities(priorities)}
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.schema_valid is True


# =========================
# 12. enum rejection: urgency / implementation_difficulty / impact
# =========================
def test_invalid_urgency_is_rejected():
    data = valid_data()
    data["containment_actions"][0]["urgency"] = "critical"
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert "containment_actions[0].urgency must be one of" in output.errors[0]


def test_invalid_implementation_difficulty_is_rejected():
    data = valid_data()
    data["permanent_actions"][0]["implementation_difficulty"] = "hard"
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert (
        "permanent_actions[0].implementation_difficulty must be one of"
        in output.errors[0]
    )


def test_invalid_impact_is_rejected():
    data = valid_data()
    data["risk_assessment"][0]["impact"] = "catastrophic"
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert "risk_assessment[0].impact must be one of" in output.errors[0]


# =========================
# 13. Provider-agnostic: agent must not import concrete providers
# =========================
def test_provider_exception_returns_needs_review_without_crash():
    agent = CountermeasurePlanningAgent(provider=BoomProvider())

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "provider_error"
    assert output.errors == ["provider blew up"]


def test_agent_defaults_to_ollama_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    agent = CountermeasurePlanningAgent()
    assert isinstance(agent.provider, OllamaProvider)


def test_agent_accepts_fugu_provider_name(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    agent = CountermeasurePlanningAgent(provider_name="fugu")
    assert isinstance(agent.provider, FuguProvider)


def test_agent_module_does_not_import_fugu_or_ollama_directly():
    import src.agents.countermeasure_planning_agent.countermeasure_planning_agent as module

    source = module.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "FuguProvider" not in text
    assert "OllamaProvider" not in text
    assert "fugu_provider" not in text
    assert "ollama_provider" not in text



# =========================================================
# New: review_reasons propagation + evidence policy review
# =========================================================
def _all_low_confidence_rca() -> RootCauseAnalysis:
    rca = make_root_cause_analysis()
    for hyp in rca.hypotheses:
        hyp["confidence"] = "low"
    return rca


def _all_low_confidence_input() -> CountermeasurePlanningInput:
    return CountermeasurePlanningInput(
        case_id="cmp-low-only",
        root_cause_analysis=_all_low_confidence_rca(),
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )


def _plan_for(rca: RootCauseAnalysis) -> dict:
    """Return a minimal schema-valid plan whose verification_plan covers all
    declared evidence/keywords for the given RCA fixture."""

    hypothesis_strings = [h["hypothesis"] for h in rca.hypotheses]
    primary = hypothesis_strings[0]

    return {
        "containment_actions": [
            {
                "action": "Quarantine suspect lot",
                "target": "lot 2025-W22",
                "purpose": "containment",
                "urgency": "high",
                "owner_candidate": "QC supervisor",
            }
        ],
        "permanent_actions": [],
        "verification_plan": [
            {
                "verification_item": "Check driver calibration record",
                "method": "audit calibration log against reference torque",
                "success_criteria": "calibration record present and within tolerance",
                "required_data": [
                    "driver calibration log",
                    "recent calibration verification record",
                ],
                "related_hypothesis": primary,
            },
            {
                "verification_item": "Confirm calibration verification interval",
                "method": "verification interval audit",
                "success_criteria": "interval calibration cadence is documented",
                "required_data": [
                    "calibration cadence policy",
                    "shift-by-shift torque distribution",
                ],
                "related_hypothesis": hypothesis_strings[1] if len(hypothesis_strings) > 1 else primary,
            },
        ],
        "risk_assessment": [
            {
                "risk": "downtime",
                "impact": "low",
                "mitigation": "off-shift window",
            }
        ],
        "priority_recommendation": [
            {"priority": 1, "action": "Quarantine", "reason": "containment"}
        ],
        "review_reasons": [],
    }


# 1. Model-emitted review_reasons must propagate into AgentOutput.
def test_model_review_reasons_propagate_to_agent_output():
    data = valid_data()
    data["review_reasons"] = [
        "Driver calibration log not yet audited",
        "Operator hypothesis is low confidence",
    ]
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.error_type == "evidence_review_required"
    for reason in data["review_reasons"]:
        assert reason in output.review_reasons
    assert output.result.review_reasons == data["review_reasons"]


# 2. RCA top-level missing_evidence not reflected => needs_review.
def test_rca_missing_evidence_not_in_verification_plan_triggers_review():
    data = valid_data()
    # Strip "driver calibration log" from required_data so the RCA-level
    # missing_evidence is no longer covered anywhere in the plan.
    for vp in data["verification_plan"]:
        vp["required_data"] = [
            r for r in vp["required_data"] if "calibration log" not in r.lower()
        ]
        # Also blank the verification_item / method that would otherwise
        # mention the calibration log keyword.
        for key in ("verification_item", "method", "success_criteria"):
            vp[key] = vp[key].replace("calibration", "general").replace(
                "Calibrated", "General"
            )

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        "RCA missing_evidence 'driver calibration log' is not reflected"
        in reason
        for reason in output.review_reasons
    )


# 3. RCA top-level missing_evidence reflected => no derived review reason.
def test_rca_missing_evidence_in_verification_plan_succeeds():
    # valid_data() already includes "calibration record" + "driver
    # calibration log" coverage matching the RCA missing_evidence fixture.
    provider = RecordingProvider(response_text(valid_data()))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "success"
    assert output.needs_review is False
    assert output.review_reasons == []


# 4. Hypothesis-level missing_evidence not reflected => needs_review.
def test_hypothesis_missing_evidence_not_reflected_triggers_review():
    data = valid_data()
    # Drop the "shift-by-shift" coverage that maps to the operator-hypothesis
    # missing_evidence.
    for vp in data["verification_plan"]:
        if vp["related_hypothesis"] == HYPOTHESIS_OPERATOR:
            vp["required_data"] = ["irrelevant data"]
            vp["verification_item"] = "Unrelated work"
            vp["method"] = "Other approach"
            vp["success_criteria"] = "n/a"

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        f"hypothesis '{HYPOTHESIS_OPERATOR}' missing_evidence "
        "'shift-by-shift torque distribution' is not reflected" in reason
        for reason in output.review_reasons
    )


# 5+6. five_why.evidence_status assumed / missing must be reflected.
@pytest.mark.parametrize("evidence_status", ["assumed", "missing"])
def test_five_why_unreflected_evidence_triggers_review(evidence_status):
    rca = make_root_cause_analysis()
    # Force a single five_why entry to the chosen status with words that the
    # plan does NOT contain anywhere.
    rca.five_why = [
        {
            "level": 1,
            "why": "Why are units below torque spec?",
            "answer": "Lubrication consistency varies between operators",
            "evidence_status": evidence_status,
        },
        {
            "level": 2,
            "why": "Why does lubrication consistency vary?",
            "answer": "Procedure document is ambiguous",
            "evidence_status": "confirmed",
        },
        {
            "level": 3,
            "why": "Why is the procedure ambiguous?",
            "answer": "Last update predates current tool model",
            "evidence_status": "confirmed",
        },
    ]
    # Also reset hypothesis missing_evidence to avoid triggering that rule.
    for hyp in rca.hypotheses:
        hyp["missing_evidence"] = []
    rca.missing_evidence = []

    cmp_input = CountermeasurePlanningInput(
        case_id="cmp-5w",
        root_cause_analysis=rca,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )

    # Plan deliberately avoids any keywords from the assumed/missing entry.
    data = valid_data()
    for vp in data["verification_plan"]:
        vp["verification_item"] = "Torque audit comparison"
        vp["method"] = "Sample audit"
        vp["success_criteria"] = "Pass rate above target"
        vp["required_data"] = ["audit results"]

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(cmp_input)

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        f"five_why level=1 evidence_status={evidence_status} is not reflected"
        in reason
        for reason in output.review_reasons
    )


# 7. low-confidence hypothesis has permanent_action but no paired verification.
def test_low_confidence_permanent_action_without_verification_triggers_review():
    data = valid_data()
    # Add a permanent_action against the low-confidence operator hypothesis
    # while ensuring no verification_plan item points at that hypothesis.
    data["permanent_actions"].append(
        {
            "action": "Roll out standardised shift handover script",
            "related_hypothesis": HYPOTHESIS_OPERATOR,
            "expected_effect": "Reduces operator-induced variance",
            "implementation_difficulty": "medium",
        }
    )
    data["verification_plan"] = [
        vp
        for vp in data["verification_plan"]
        if vp["related_hypothesis"] != HYPOTHESIS_OPERATOR
    ]

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        f"permanent_action for low-confidence hypothesis '{HYPOTHESIS_OPERATOR}'"
        in reason
        for reason in output.review_reasons
    )


# 8. Low-confidence-only RCA: verification-only plan (no permanent_actions) is OK
#    as long as review_reasons explains the low-confidence deferral.
def test_low_confidence_only_rca_allows_verification_only_plan():
    rca = _all_low_confidence_rca()
    cmp_input = CountermeasurePlanningInput(
        case_id="cmp-low-only",
        root_cause_analysis=rca,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )

    data = _plan_for(rca)
    data["review_reasons"] = [
        "All RCA hypotheses are low confidence; permanent actions deferred "
        "pending verification."
    ]

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(cmp_input)

    # The plan is structurally valid; the only review_reasons are the model's,
    # which themselves push the run into needs_review.
    assert output.parse_success is True
    assert output.schema_valid is True
    assert output.status == "needs_review"
    assert output.needs_review is True
    assert output.review_reasons == data["review_reasons"]
    assert all(
        "permanent_actions is empty" not in reason for reason in output.review_reasons
    )


def test_low_confidence_only_rca_requires_review_reason_when_permanent_empty():
    """If the model leaves review_reasons empty under low-confidence-only
    hypotheses, the agent must derive a review reason on its behalf."""

    rca = _all_low_confidence_rca()
    cmp_input = CountermeasurePlanningInput(
        case_id="cmp-low-only-missing-reason",
        root_cause_analysis=rca,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
    )

    data = _plan_for(rca)
    data["review_reasons"] = []  # model forgets to explain

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(cmp_input)

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        "permanent_actions is empty under low-confidence-only hypotheses"
        in reason
        for reason in output.review_reasons
    )


# 9. medium/high confidence hypothesis exists but permanent_actions is empty.
def test_non_low_confidence_hypothesis_requires_permanent_actions():
    data = valid_data()
    data["permanent_actions"] = []

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())  # RCA fixture has a HIGH-confidence hyp

    assert output.status == "needs_review"
    assert output.error_type == "evidence_review_required"
    assert any(
        "permanent_actions is empty but non-low-confidence hypotheses exist"
        in reason
        for reason in output.review_reasons
    )


# 10. Unknown related_hypothesis is still rejected at structural level.
def test_unknown_related_hypothesis_is_schema_validation_error():
    data = valid_data()
    data["permanent_actions"][0]["related_hypothesis"] = "totally invented one"

    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert (
        "related_hypothesis must match one of the input RCA hypotheses"
        in output.errors[0]
    )


# Structural: review_reasons type must be list[str].
@pytest.mark.parametrize(
    ("bad_review_reasons", "expected_error"),
    [
        ("not a list", "review_reasons must be a list"),
        ([1, 2, 3], "review_reasons[0] must be a string"),
        ([None], "review_reasons[0] must be a string"),
    ],
)
def test_review_reasons_must_be_list_of_strings(bad_review_reasons, expected_error):
    data = {**valid_data(), "review_reasons": bad_review_reasons}
    provider = RecordingProvider(response_text(data))
    agent = CountermeasurePlanningAgent(provider=provider)

    output = agent.run(make_input())

    assert output.status == "needs_review"
    assert output.error_type == "schema_validation_error"
    assert expected_error in output.errors[0]
