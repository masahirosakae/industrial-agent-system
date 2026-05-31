import json

import pytest

from src.agents.countermeasure_planning_agent import (
    CountermeasurePlan,
    CountermeasurePlanningAgentOutput,
)
from src.agents.quality_evaluation_agent import (
    QualityEvaluationAgentOutput,
    QualityEvaluationResult,
)
from src.agents.quality_issue_analysis_agent import (
    QualityIssueAgentOutput,
    QualityIssueAnalysis,
)
from src.agents.root_cause_analysis_agent import (
    RootCauseAgentOutput,
    RootCauseAnalysis,
)
from src.llm.base import LLMProvider, LLMResponse
from src.workflows import (
    QUALITY_WORKFLOW_NAME,
    QualityWorkflowInput,
    QualityWorkflowResult,
    run_quality_workflow,
    workflow_result_to_dict,
)


# -------------------------
# Lightweight test doubles
# -------------------------
class StubProvider(LLMProvider):
    """A LLMProvider stub. Mock agents do not actually call it, but the
    workflow needs *some* provider object so it can pass it to default
    agents and so the result records a provider_name."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:  # pragma: no cover - never invoked
        return LLMResponse(text="{}", model="test-model", provider="test")


class MockAgent:
    """Minimal stand-in: mirrors the agent interface (``agent_name``,
    ``run(input)``) without invoking any LLM."""

    def __init__(self, agent_name: str, output):
        self.agent_name = agent_name
        self._output = output
        self.calls: list = []

    def run(self, agent_input):
        self.calls.append(agent_input)
        return self._output


class BoomMockAgent:
    """Raises an unexpected exception when ``run`` is invoked."""

    def __init__(self, agent_name: str, exc: Exception):
        self.agent_name = agent_name
        self._exc = exc
        self.calls: list = []

    def run(self, agent_input):
        self.calls.append(agent_input)
        raise self._exc


# -------------------------
# Canonical inputs / outputs
# -------------------------
HYPOTHESIS_DRIVER = "DC driver transducer is drifting low"


def make_workflow_input(case_id: str = "qw-001") -> QualityWorkflowInput:
    return QualityWorkflowInput(
        case_id=case_id,
        process_name="screw_fastening",
        product_or_part="aluminum housing M4",
        defect_mode="loose torque on assembled units",
        observed_symptoms=["torque audit shows 12% units below spec"],
        process_conditions={"target_torque_Nm": 1.2},
        known_constraints=["screw spec frozen"],
        available_data=["torque trace log"],
    )


def make_quality_issue_analysis() -> QualityIssueAnalysis:
    return QualityIssueAnalysis(
        issue_summary="Loose torque traced to driver calibration drift.",
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


def make_root_cause_analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        problem_statement="Torque audit shows 12% units below spec.",
        facts=[
            {
                "fact": "post-assembly torque audit shows 12% units below spec",
                "source": "qia.issue_summary",
            }
        ],
        assumptions=[],
        hypotheses=[
            {
                "hypothesis": HYPOTHESIS_DRIVER,
                "confidence": "high",
                "supporting_facts": ["torque audit 12% under spec"],
                "missing_evidence": [],
            }
        ],
        five_why=[
            {
                "level": 1,
                "why": "w1",
                "answer": "a1",
                "evidence_status": "confirmed",
            },
            {
                "level": 2,
                "why": "w2",
                "answer": "a2",
                "evidence_status": "confirmed",
            },
            {
                "level": 3,
                "why": "w3",
                "answer": "a3",
                "evidence_status": "confirmed",
            },
        ],
        fta_tree={
            "top_event": "Loose torque on assembled units",
            "branches": [{"category": "equipment", "causes": ["driver drift"]}],
        },
        missing_evidence=[],
    )


def make_countermeasure_plan() -> CountermeasurePlan:
    return CountermeasurePlan(
        containment_actions=[
            {
                "action": "100% torque audit",
                "target": "lot 2025-W22",
                "purpose": "containment",
                "urgency": "high",
                "owner_candidate": "QC supervisor",
            }
        ],
        permanent_actions=[
            {
                "action": "Tighten driver calibration cadence",
                "related_hypothesis": HYPOTHESIS_DRIVER,
                "expected_effect": "Driver torque within +/- 5%",
                "implementation_difficulty": "medium",
            }
        ],
        verification_plan=[
            {
                "verification_item": "Driver torque vs reference",
                "method": "reference transducer test",
                "success_criteria": "All within +/- 5%",
                "required_data": ["calibration record"],
                "related_hypothesis": HYPOTHESIS_DRIVER,
            }
        ],
        risk_assessment=[
            {"risk": "downtime", "impact": "low", "mitigation": "off-shift"}
        ],
        priority_recommendation=[
            {"priority": 1, "action": "audit", "reason": "containment"}
        ],
        review_reasons=[],
    )


def make_quality_evaluation_result(overall: str = "accepted") -> QualityEvaluationResult:
    return QualityEvaluationResult(
        evidence_sufficiency={"score": 0.85, "assessment": "ok", "insufficient_items": []},
        hallucination_risk={"risk_level": "low", "suspected_items": []},
        rca_cmp_consistency={"score": 0.9, "assessment": "ok", "inconsistencies": []},
        countermeasure_quality={"score": 0.8, "assessment": "ok", "weaknesses": []},
        verification_quality={"score": 0.85, "assessment": "ok", "missing_verifications": []},
        approval_readiness={"score": 0.85, "assessment": "ok"},
        overall_judgement=overall,
        review_reasons=[] if overall == "accepted" else ["analyst requested review"],
        recommended_next_steps=["run the audit"],
    )


def success_qia_output() -> QualityIssueAgentOutput:
    return QualityIssueAgentOutput(
        case_id="qw-001",
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


def needs_review_qia_output() -> QualityIssueAgentOutput:
    # NOTE: QualityIssueAgentOutput does not (yet) expose ``review_reasons``;
    # the workflow's _collect_review_reasons falls back to ``errors`` for QIA.
    return QualityIssueAgentOutput(
        case_id="qw-001",
        agent_name="quality_issue_analysis_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["LLM output could not be parsed"],
        notes=["fallback"],
        needs_review=True,
        parse_success=False,
        schema_valid=False,
        error_type="parse_error",
        raw_output="not json",
    )


def success_rca_output() -> RootCauseAgentOutput:
    return RootCauseAgentOutput(
        case_id="qw-001",
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


def needs_review_rca_output() -> RootCauseAgentOutput:
    return RootCauseAgentOutput(
        case_id="qw-001",
        agent_name="root_cause_analysis_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["schema validation failed"],
        notes=["fallback"],
        needs_review=True,
        review_reasons=["schema_validation_error: ..."],
        parse_success=True,
        schema_valid=False,
        error_type="schema_validation_error",
        raw_output="{}",
    )


def success_cmp_output() -> CountermeasurePlanningAgentOutput:
    return CountermeasurePlanningAgentOutput(
        case_id="qw-001",
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


def needs_review_cmp_output() -> CountermeasurePlanningAgentOutput:
    return CountermeasurePlanningAgentOutput(
        case_id="qw-001",
        agent_name="countermeasure_planning_agent",
        status="needs_review",
        result=None,
        confidence=0.0,
        errors=["evidence cross-check failed"],
        notes=["fallback"],
        needs_review=True,
        review_reasons=["evidence_review_required: missing_evidence not covered"],
        parse_success=True,
        schema_valid=True,
        error_type="evidence_review_required",
        raw_output="{}",
    )


def success_qea_output() -> QualityEvaluationAgentOutput:
    return QualityEvaluationAgentOutput(
        case_id="qw-001",
        status="success",
        result=make_quality_evaluation_result("accepted"),
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


def needs_review_qea_output() -> QualityEvaluationAgentOutput:
    return QualityEvaluationAgentOutput(
        case_id="qw-001",
        status="needs_review",
        result=make_quality_evaluation_result("needs_review"),
        confidence=0.5,
        errors=[],
        notes=["policy review required"],
        needs_review=True,
        review_reasons=["analyst requested review"],
        parse_success=True,
        schema_valid=True,
        error_type="policy_review_required",
        raw_output="{}",
    )


# =========================================================
# 1. All agents success -> workflow success
# =========================================================
def test_workflow_success_path_runs_all_four_agents():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert isinstance(result, QualityWorkflowResult)
    assert result.workflow_name == QUALITY_WORKFLOW_NAME
    assert result.status == "success"
    assert result.final_judgement == "accepted"
    assert result.stopped_at is None
    assert result.review_reasons == []
    assert result.provider_name == "stub"
    assert list(result.steps) == [
        "quality_issue_analysis_agent",
        "root_cause_analysis_agent",
        "countermeasure_planning_agent",
        "quality_evaluation_agent",
    ]
    assert all(agent.calls for agent in (qia, rca, cmp_, qea))


# =========================================================
# 2. QIA needs_review -> stop, no downstream calls
# =========================================================
def test_workflow_stops_when_qia_returns_needs_review():
    qia = MockAgent("quality_issue_analysis_agent", needs_review_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.status == "needs_review"
    assert result.final_judgement == "needs_review"
    assert result.stopped_at == "quality_issue_analysis_agent"
    assert list(result.steps) == ["quality_issue_analysis_agent"]
    assert rca.calls == []
    assert cmp_.calls == []
    assert qea.calls == []
    assert any(
        "quality_issue_analysis_agent" in reason for reason in result.review_reasons
    )


# =========================================================
# 3. RCA needs_review -> stop after RCA
# =========================================================
def test_workflow_stops_when_rca_returns_needs_review():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", needs_review_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.status == "needs_review"
    assert result.stopped_at == "root_cause_analysis_agent"
    assert list(result.steps) == [
        "quality_issue_analysis_agent",
        "root_cause_analysis_agent",
    ]
    assert cmp_.calls == []
    assert qea.calls == []


# =========================================================
# 4. CMP needs_review -> stop after CMP
# =========================================================
def test_workflow_stops_when_cmp_returns_needs_review():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", needs_review_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.status == "needs_review"
    assert result.stopped_at == "countermeasure_planning_agent"
    assert list(result.steps) == [
        "quality_issue_analysis_agent",
        "root_cause_analysis_agent",
        "countermeasure_planning_agent",
    ]
    assert qea.calls == []


# =========================================================
# 5. QEA needs_review -> workflow needs_review
# =========================================================
def test_workflow_marks_needs_review_when_qea_judges_needs_review():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", needs_review_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.status == "needs_review"
    assert result.final_judgement == "needs_review"
    assert result.stopped_at == "quality_evaluation_agent"
    assert list(result.steps) == [
        "quality_issue_analysis_agent",
        "root_cause_analysis_agent",
        "countermeasure_planning_agent",
        "quality_evaluation_agent",
    ]


# QEA reports status="success" but overall_judgement="needs_review" -> also stops.
def test_workflow_needs_review_when_qea_success_but_overall_judgement_says_review():
    qea_output = success_qea_output()
    qea_output.result.overall_judgement = "needs_review"

    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", qea_output)

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.status == "needs_review"
    assert result.stopped_at == "quality_evaluation_agent"


# =========================================================
# 6. Unexpected exception -> safe needs_review envelope
# =========================================================
def test_workflow_treats_unexpected_exception_as_needs_review():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = BoomMockAgent(
        "root_cause_analysis_agent", RuntimeError("rca exploded mid-run")
    )

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
    )

    # Safe side: never raises out of run_quality_workflow, never returns
    # "failure" silently.
    assert result.status == "needs_review"
    assert result.final_judgement == "needs_review"
    assert result.stopped_at == "root_cause_analysis_agent"
    # QIA result is preserved in steps even when downstream blows up.
    assert "quality_issue_analysis_agent" in result.steps
    assert "root_cause_analysis_agent" not in result.steps
    assert any(
        "unexpected_exception" in reason and "rca exploded mid-run" in reason
        for reason in result.review_reasons
    )


# =========================================================
# 7. Steps dict only contains executed agents
# =========================================================
def test_steps_only_contains_executed_agents_after_early_stop():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", needs_review_rca_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
    )

    assert set(result.steps) == {
        "quality_issue_analysis_agent",
        "root_cause_analysis_agent",
    }
    assert "countermeasure_planning_agent" not in result.steps
    assert "quality_evaluation_agent" not in result.steps


# =========================================================
# 8. provider_name is recorded on the result
# =========================================================
def test_provider_name_is_recorded_explicitly():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="fugu",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    assert result.provider_name == "fugu"


def test_provider_name_is_derived_from_provider_class_when_not_given():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    # StubProvider -> "stub" via the workflow's class-name heuristic.
    assert result.provider_name == "stub"


# =========================================================
# Workflow result is JSON-serialisable
# =========================================================
def test_workflow_result_round_trips_through_json():
    qia = MockAgent("quality_issue_analysis_agent", success_qia_output())
    rca = MockAgent("root_cause_analysis_agent", success_rca_output())
    cmp_ = MockAgent("countermeasure_planning_agent", success_cmp_output())
    qea = MockAgent("quality_evaluation_agent", success_qea_output())

    result = run_quality_workflow(
        make_workflow_input(),
        provider=StubProvider(),
        provider_name="stub",
        qia_agent=qia,
        rca_agent=rca,
        cmp_agent=cmp_,
        qea_agent=qea,
    )

    payload = workflow_result_to_dict(result)
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["status"] == "success"
    assert decoded["final_judgement"] == "accepted"
    assert decoded["provider_name"] == "stub"
    assert decoded["steps"]["quality_evaluation_agent"]["result"][
        "overall_judgement"
    ] == "accepted"
