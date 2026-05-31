from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# NOTE: The workflow depends only on the abstract LLMProvider contract and on
# each agent's public schema dataclasses + its own ``from_*_output`` input
# gate. Concrete provider implementations must never be imported here; the
# provider-agnostic test guards this contract.
from src.agents.countermeasure_planning_agent import (
    CountermeasurePlanningAgent,
    CountermeasurePlanningInput,
)
from src.agents.quality_evaluation_agent import (
    QualityEvaluationAgent,
    QualityEvaluationInput,
)
from src.agents.quality_issue_analysis_agent import (
    QualityIssueAnalysisAgent,
    QualityIssueInput,
)
from src.agents.root_cause_analysis_agent import (
    RootCauseAnalysisAgent,
    RootCauseAnalysisInput,
)
from src.llm.base import LLMProvider
from src.llm.factory import create_llm_provider


QUALITY_WORKFLOW_NAME = "phase1_quality_workflow"


QualityWorkflowStatus = Literal["success", "needs_review", "failure"]
QualityWorkflowJudgement = Literal["accepted", "needs_review"]


# -------------------------
# Input / Output schema
# -------------------------
@dataclass
class QualityWorkflowInput:
    """Workflow-level input that mirrors QualityIssueAnalysisAgent's input.

    Kept as a 1:1 mapping so the workflow does not silently transform fields
    on behalf of the caller; the QIA contract remains the single source of
    truth for what a "quality issue case" looks like in Phase 1.
    """

    case_id: str
    process_name: str
    product_or_part: str
    defect_mode: str
    observed_symptoms: list[str] = field(default_factory=list)
    process_conditions: dict = field(default_factory=dict)
    known_constraints: list[str] = field(default_factory=list)
    available_data: list[str] = field(default_factory=list)


@dataclass
class QualityWorkflowResult:
    workflow_name: str
    case_id: str
    status: QualityWorkflowStatus
    final_judgement: QualityWorkflowJudgement
    stopped_at: str | None
    review_reasons: list[str]
    steps: dict[str, Any]
    provider_name: str


# -------------------------
# Orchestrator
# -------------------------
def run_quality_workflow(
    workflow_input: QualityWorkflowInput,
    *,
    provider: LLMProvider | None = None,
    provider_name: str | None = None,
    model: str = "qwen2.5:1.5b",
    qia_agent: QualityIssueAnalysisAgent | None = None,
    rca_agent: RootCauseAnalysisAgent | None = None,
    cmp_agent: CountermeasurePlanningAgent | None = None,
    qea_agent: QualityEvaluationAgent | None = None,
) -> QualityWorkflowResult:
    """Run the Phase 1 Quality Workflow end-to-end.

    Execution order: QIA -> RCA -> CMP -> QEA. Each stage's
    ``status != "success"`` halts the workflow immediately; the next agent is
    not even constructed, so a needs_review or failed earlier stage cannot
    leak into a later stage's input gate.

    Provider is resolved once and shared across every agent that the
    workflow instantiates. Callers may also inject pre-built agents (mainly
    for testing), in which case the shared-provider construction is skipped
    for those slots.
    """

    resolved_provider, resolved_provider_name = _resolve_provider(
        provider=provider, provider_name=provider_name, model=model
    )

    steps: dict[str, Any] = {}

    # ------------------------------
    # Step 1: QualityIssueAnalysisAgent
    # ------------------------------
    qia_input = _build_qia_input(workflow_input)
    try:
        qia = qia_agent or QualityIssueAnalysisAgent(provider=resolved_provider)
        qia_output = qia.run(qia_input)
    except Exception as e:
        return _unexpected_exception_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=QualityIssueAnalysisAgent.agent_name,
            steps=steps,
            error=e,
        )
    steps[QualityIssueAnalysisAgent.agent_name] = qia_output
    if qia_output.status != "success":
        return _stopped_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=QualityIssueAnalysisAgent.agent_name,
            steps=steps,
            review_reasons=_collect_review_reasons(qia_output),
        )

    # ------------------------------
    # Step 2: RootCauseAnalysisAgent
    # ------------------------------
    try:
        rca_input = RootCauseAnalysisInput.from_quality_issue_output(
            qia_output,
            process_name=workflow_input.process_name,
            product_or_part=workflow_input.product_or_part,
        )
        rca = rca_agent or RootCauseAnalysisAgent(provider=resolved_provider)
        rca_output = rca.run(rca_input)
    except Exception as e:
        return _unexpected_exception_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=RootCauseAnalysisAgent.agent_name,
            steps=steps,
            error=e,
        )
    steps[RootCauseAnalysisAgent.agent_name] = rca_output
    if rca_output.status != "success":
        return _stopped_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=RootCauseAnalysisAgent.agent_name,
            steps=steps,
            review_reasons=_collect_review_reasons(rca_output),
        )

    # ------------------------------
    # Step 3: CountermeasurePlanningAgent
    # ------------------------------
    try:
        cmp_input = CountermeasurePlanningInput.from_root_cause_output(
            rca_output,
            process_name=workflow_input.process_name,
            product_or_part=workflow_input.product_or_part,
        )
        cmp = cmp_agent or CountermeasurePlanningAgent(provider=resolved_provider)
        cmp_output = cmp.run(cmp_input)
    except Exception as e:
        return _unexpected_exception_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=CountermeasurePlanningAgent.agent_name,
            steps=steps,
            error=e,
        )
    steps[CountermeasurePlanningAgent.agent_name] = cmp_output
    if cmp_output.status != "success":
        return _stopped_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=CountermeasurePlanningAgent.agent_name,
            steps=steps,
            review_reasons=_collect_review_reasons(cmp_output),
        )

    # ------------------------------
    # Step 4: QualityEvaluationAgent
    # ------------------------------
    try:
        qea_input = QualityEvaluationInput.from_previous_outputs(
            rca_output,
            cmp_output,
            process_name=workflow_input.process_name,
            product_or_part=workflow_input.product_or_part,
        )
        qea = qea_agent or QualityEvaluationAgent(provider=resolved_provider)
        qea_output = qea.run(qea_input)
    except Exception as e:
        return _unexpected_exception_result(
            workflow_input=workflow_input,
            provider_name=resolved_provider_name,
            stopped_at=QualityEvaluationAgent.agent_name,
            steps=steps,
            error=e,
        )
    steps[QualityEvaluationAgent.agent_name] = qea_output

    if (
        qea_output.status == "success"
        and qea_output.result is not None
        and qea_output.result.overall_judgement == "accepted"
    ):
        return QualityWorkflowResult(
            workflow_name=QUALITY_WORKFLOW_NAME,
            case_id=workflow_input.case_id,
            status="success",
            final_judgement="accepted",
            stopped_at=None,
            review_reasons=[],
            steps=steps,
            provider_name=resolved_provider_name,
        )

    return _stopped_result(
        workflow_input=workflow_input,
        provider_name=resolved_provider_name,
        stopped_at=QualityEvaluationAgent.agent_name,
        steps=steps,
        review_reasons=_collect_review_reasons(qea_output),
    )


# -------------------------
# Helpers
# -------------------------
def _build_qia_input(workflow_input: QualityWorkflowInput) -> QualityIssueInput:
    return QualityIssueInput(
        case_id=workflow_input.case_id,
        process_name=workflow_input.process_name,
        product_or_part=workflow_input.product_or_part,
        defect_mode=workflow_input.defect_mode,
        observed_symptoms=list(workflow_input.observed_symptoms),
        process_conditions=dict(workflow_input.process_conditions),
        known_constraints=list(workflow_input.known_constraints),
        available_data=list(workflow_input.available_data),
    )


def _resolve_provider(
    provider: LLMProvider | None,
    provider_name: str | None,
    model: str,
) -> tuple[LLMProvider, str]:
    if provider is None:
        provider = create_llm_provider(
            provider_name=provider_name,
            ollama_model=model,
        )
    resolved_name = provider_name or _derive_provider_name(provider)
    return provider, resolved_name


def _derive_provider_name(provider: LLMProvider) -> str:
    class_name = type(provider).__name__
    if class_name.endswith("Provider"):
        return class_name[: -len("Provider")].lower()
    return class_name.lower()


def _collect_review_reasons(agent_output) -> list[str]:
    """Prefix every reason / error with the agent name for workflow-level
    traceability."""
    reasons: list[str] = []
    agent_name = getattr(agent_output, "agent_name", "unknown_agent")
    for reason in getattr(agent_output, "review_reasons", None) or []:
        if isinstance(reason, str):
            reasons.append(f"{agent_name}: {reason}")
    for error in getattr(agent_output, "errors", None) or []:
        reasons.append(f"{agent_name} error: {error}")
    if not reasons:
        error_type = getattr(agent_output, "error_type", None)
        if error_type:
            reasons.append(f"{agent_name}: error_type={error_type}")
        else:
            reasons.append(
                f"{agent_name}: status={getattr(agent_output, 'status', 'unknown')!r}"
            )
    return reasons


def _stopped_result(
    workflow_input: QualityWorkflowInput,
    provider_name: str,
    stopped_at: str,
    steps: dict[str, Any],
    review_reasons: list[str],
) -> QualityWorkflowResult:
    return QualityWorkflowResult(
        workflow_name=QUALITY_WORKFLOW_NAME,
        case_id=workflow_input.case_id,
        status="needs_review",
        final_judgement="needs_review",
        stopped_at=stopped_at,
        review_reasons=review_reasons,
        steps=steps,
        provider_name=provider_name,
    )


def _unexpected_exception_result(
    workflow_input: QualityWorkflowInput,
    provider_name: str,
    stopped_at: str,
    steps: dict[str, Any],
    error: Exception,
) -> QualityWorkflowResult:
    # Safe-side policy: an unexpected exception during orchestration is
    # mapped to needs_review (not failure) so the workflow always returns a
    # well-formed envelope that downstream tooling can JSON-serialise.
    return QualityWorkflowResult(
        workflow_name=QUALITY_WORKFLOW_NAME,
        case_id=workflow_input.case_id,
        status="needs_review",
        final_judgement="needs_review",
        stopped_at=stopped_at,
        review_reasons=[
            f"{stopped_at}: unexpected_exception "
            f"{type(error).__name__}: {error}"
        ],
        steps=steps,
        provider_name=provider_name,
    )


def workflow_result_to_dict(result: QualityWorkflowResult) -> dict:
    """Convert a QualityWorkflowResult into a JSON-serialisable dict.

    Each agent's AgentOutput is converted with ``asdict`` so nested
    dataclasses survive the trip into ``json.dumps``.
    """
    return {
        "workflow_name": result.workflow_name,
        "case_id": result.case_id,
        "status": result.status,
        "final_judgement": result.final_judgement,
        "stopped_at": result.stopped_at,
        "review_reasons": list(result.review_reasons),
        "provider_name": result.provider_name,
        "steps": {
            name: _agent_output_to_dict(output)
            for name, output in result.steps.items()
        },
    }


def _agent_output_to_dict(output: Any) -> Any:
    try:
        return asdict(output)
    except TypeError:
        return output
