import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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
    """Workflow-level input that mirrors QualityIssueAnalysisAgent's input."""

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
    # Run-level timing metadata (populated for every run, including stopped runs).
    started_at: str = ""
    finished_at: str = ""
    total_latency_seconds: float = 0.0
    step_latencies: dict[str, float] = field(default_factory=dict)


# -------------------------
# Common success gate
# -------------------------
def is_successful_agent_output(output: Any) -> bool:
    """Shared per-step success gate.

    An agent output is treated as trusted only when **all** of the following
    are true:

    - ``status == "success"``
    - ``needs_review is False``
    - ``parse_success is True``
    - ``schema_valid is True``
    - ``result is not None``

    The workflow uses this helper to decide whether to advance to the next
    step. A ``status="success"`` value alone is not enough: the same five
    conditions are also what each downstream agent's ``from_*_output`` gate
    enforces, so the workflow refuses to call those gates if any field is
    inconsistent.
    """
    return (
        getattr(output, "status", None) == "success"
        and getattr(output, "needs_review", True) is False
        and getattr(output, "parse_success", False) is True
        and getattr(output, "schema_valid", False) is True
        and getattr(output, "result", None) is not None
    )


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

    Execution order: QIA -> RCA -> CMP -> QEA. Each stage runs through:

    1. ``run`` is invoked, wrapped in a per-step latency timer.
    2. The output is checked with :func:`is_successful_agent_output` (the
       shared success gate). Any non-success state stops the workflow.
    3. The output's ``case_id`` is compared against the workflow input. A
       mismatch stops the workflow with a dedicated review_reasons entry.

    Only after both checks pass does the workflow build the next agent's
    input (which itself runs an independent trust gate). This prevents any
    needs_review or inconsistent earlier stage from leaking into a later
    stage's input.

    Provider is resolved once and shared across every agent that the
    workflow instantiates. Callers may also inject pre-built agents (mainly
    for testing), in which case the shared-provider construction is skipped
    for those slots.
    """

    resolved_provider, resolved_provider_name = _resolve_provider(
        provider=provider, provider_name=provider_name, model=model
    )

    steps: dict[str, Any] = {}
    step_latencies: dict[str, float] = {}

    started_at = _iso_now()
    workflow_started = perf_counter()

    def _finalize_stopped(stopped_at: str, review_reasons: list[str]) -> QualityWorkflowResult:
        return QualityWorkflowResult(
            workflow_name=QUALITY_WORKFLOW_NAME,
            case_id=workflow_input.case_id,
            status="needs_review",
            final_judgement="needs_review",
            stopped_at=stopped_at,
            review_reasons=review_reasons,
            steps=steps,
            provider_name=resolved_provider_name,
            started_at=started_at,
            finished_at=_iso_now(),
            total_latency_seconds=round(perf_counter() - workflow_started, 6),
            step_latencies=dict(step_latencies),
        )

    def _finalize_unexpected_exception(stopped_at: str, error: Exception) -> QualityWorkflowResult:
        # Safe-side policy: unexpected orchestration exceptions are mapped to
        # needs_review (never failure) so the workflow always returns a
        # well-formed envelope.
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
            provider_name=resolved_provider_name,
            started_at=started_at,
            finished_at=_iso_now(),
            total_latency_seconds=round(perf_counter() - workflow_started, 6),
            step_latencies=dict(step_latencies),
        )

    # ------------------------------
    # Step 1: QualityIssueAnalysisAgent
    # ------------------------------
    qia_input = _build_qia_input(workflow_input)
    try:
        qia = qia_agent or QualityIssueAnalysisAgent(provider=resolved_provider)
        qia_output = _run_step(
            QualityIssueAnalysisAgent.agent_name, qia, qia_input, step_latencies
        )
    except Exception as e:
        return _finalize_unexpected_exception(
            QualityIssueAnalysisAgent.agent_name, e
        )
    steps[QualityIssueAnalysisAgent.agent_name] = qia_output

    if not is_successful_agent_output(qia_output):
        return _finalize_stopped(
            QualityIssueAnalysisAgent.agent_name,
            _collect_review_reasons(qia_output),
        )
    mismatch = _case_id_mismatch(
        qia_output, workflow_input, QualityIssueAnalysisAgent.agent_name
    )
    if mismatch:
        return _finalize_stopped(QualityIssueAnalysisAgent.agent_name, [mismatch])

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
        rca_output = _run_step(
            RootCauseAnalysisAgent.agent_name, rca, rca_input, step_latencies
        )
    except Exception as e:
        return _finalize_unexpected_exception(
            RootCauseAnalysisAgent.agent_name, e
        )
    steps[RootCauseAnalysisAgent.agent_name] = rca_output

    if not is_successful_agent_output(rca_output):
        return _finalize_stopped(
            RootCauseAnalysisAgent.agent_name,
            _collect_review_reasons(rca_output),
        )
    mismatch = _case_id_mismatch(
        rca_output, workflow_input, RootCauseAnalysisAgent.agent_name
    )
    if mismatch:
        return _finalize_stopped(RootCauseAnalysisAgent.agent_name, [mismatch])

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
        cmp_output = _run_step(
            CountermeasurePlanningAgent.agent_name, cmp, cmp_input, step_latencies
        )
    except Exception as e:
        return _finalize_unexpected_exception(
            CountermeasurePlanningAgent.agent_name, e
        )
    steps[CountermeasurePlanningAgent.agent_name] = cmp_output

    if not is_successful_agent_output(cmp_output):
        return _finalize_stopped(
            CountermeasurePlanningAgent.agent_name,
            _collect_review_reasons(cmp_output),
        )
    mismatch = _case_id_mismatch(
        cmp_output, workflow_input, CountermeasurePlanningAgent.agent_name
    )
    if mismatch:
        return _finalize_stopped(CountermeasurePlanningAgent.agent_name, [mismatch])

    # ------------------------------
    # Step 4: QualityEvaluationAgent
    # ------------------------------
    # Defensive: RCA and CMP case_ids were verified individually above, so
    # they are necessarily equal here. Re-check explicitly so the QEA input
    # construction is robust to future refactors.
    if rca_output.case_id != cmp_output.case_id:
        return _finalize_stopped(
            QualityEvaluationAgent.agent_name,
            [
                "quality_evaluation_agent: case_id mismatch between "
                f"RCA ({rca_output.case_id!r}) and CMP ({cmp_output.case_id!r})"
            ],
        )

    try:
        qea_input = QualityEvaluationInput.from_previous_outputs(
            rca_output,
            cmp_output,
            process_name=workflow_input.process_name,
            product_or_part=workflow_input.product_or_part,
        )
        qea = qea_agent or QualityEvaluationAgent(provider=resolved_provider)
        qea_output = _run_step(
            QualityEvaluationAgent.agent_name, qea, qea_input, step_latencies
        )
    except Exception as e:
        return _finalize_unexpected_exception(
            QualityEvaluationAgent.agent_name, e
        )
    steps[QualityEvaluationAgent.agent_name] = qea_output

    mismatch = _case_id_mismatch(
        qea_output, workflow_input, QualityEvaluationAgent.agent_name
    )
    if mismatch:
        return _finalize_stopped(QualityEvaluationAgent.agent_name, [mismatch])

    if (
        is_successful_agent_output(qea_output)
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
            started_at=started_at,
            finished_at=_iso_now(),
            total_latency_seconds=round(perf_counter() - workflow_started, 6),
            step_latencies=dict(step_latencies),
        )

    return _finalize_stopped(
        QualityEvaluationAgent.agent_name,
        _collect_review_reasons(qea_output),
    )


# -------------------------
# Helpers
# -------------------------
def _run_step(
    agent_name: str,
    agent: Any,
    agent_input: Any,
    step_latencies: dict[str, float],
) -> Any:
    """Invoke ``agent.run(agent_input)`` while recording its latency.

    Latency is recorded even when ``run`` raises, so trace logs always reflect
    the step that was actually attempted.
    """
    started = perf_counter()
    try:
        return agent.run(agent_input)
    finally:
        step_latencies[agent_name] = round(perf_counter() - started, 6)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _case_id_mismatch(
    agent_output: Any,
    workflow_input: QualityWorkflowInput,
    agent_name: str,
) -> str | None:
    actual = getattr(agent_output, "case_id", None)
    if actual == workflow_input.case_id:
        return None
    return (
        f"{agent_name}: case_id mismatch "
        f"(expected {workflow_input.case_id!r}, got {actual!r})"
    )


def workflow_result_to_dict(result: QualityWorkflowResult) -> dict:
    """Convert a QualityWorkflowResult into a JSON-serialisable dict."""
    return {
        "workflow_name": result.workflow_name,
        "case_id": result.case_id,
        "status": result.status,
        "final_judgement": result.final_judgement,
        "stopped_at": result.stopped_at,
        "review_reasons": list(result.review_reasons),
        "provider_name": result.provider_name,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "total_latency_seconds": result.total_latency_seconds,
        "step_latencies": dict(result.step_latencies),
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


# -------------------------
# Trace logging (one JSONL line per workflow run)
# -------------------------
def build_quality_workflow_trace(
    result: QualityWorkflowResult,
    *,
    run_id: str | None = None,
    case_version: str | None = None,
    model: str | None = None,
    include_raw_output: bool = False,
) -> dict:
    """Build a compact, evaluation-friendly trace dict for one workflow run.

    Phase 1 default is ``include_raw_output=False``: raw LLM text is not
    embedded in the trace so the file can be shared with reviewers without
    leaking model outputs verbatim. Set ``include_raw_output=True`` when
    debugging a single run locally.

    The returned structure is guaranteed to be JSON-serialisable.
    """
    steps_trace: list[dict] = []
    for name, output in result.steps.items():
        step_entry = {
            "agent_name": name,
            "status": getattr(output, "status", None),
            "needs_review": getattr(output, "needs_review", None),
            "parse_success": getattr(output, "parse_success", None),
            "schema_valid": getattr(output, "schema_valid", None),
            "error_type": getattr(output, "error_type", None),
            "latency_seconds": result.step_latencies.get(name),
            "review_reasons": list(
                getattr(output, "review_reasons", None) or []
            ),
        }
        if include_raw_output:
            step_entry["raw_output"] = getattr(output, "raw_output", "")
        steps_trace.append(step_entry)

    return {
        "run_id": run_id or str(uuid.uuid4()),
        "case_id": result.case_id,
        "case_version": case_version,
        "provider_name": result.provider_name,
        "model": model,
        "workflow_name": result.workflow_name,
        "workflow_status": result.status,
        "final_judgement": result.final_judgement,
        "stopped_at": result.stopped_at,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "total_latency_seconds": result.total_latency_seconds,
        "review_reasons": list(result.review_reasons),
        "steps": steps_trace,
        "qea_scores": _extract_qea_scores(
            result.steps.get("quality_evaluation_agent")
        ),
    }


def append_quality_workflow_trace(
    result: QualityWorkflowResult,
    path: Path,
    *,
    run_id: str | None = None,
    case_version: str | None = None,
    model: str | None = None,
    include_raw_output: bool = False,
) -> dict:
    """Append a single trace entry as a JSONL line to ``path``.

    Returns the trace dict that was written so callers can inspect or print
    it without re-reading the file.
    """
    trace = build_quality_workflow_trace(
        result,
        run_id=run_id,
        case_version=case_version,
        model=model,
        include_raw_output=include_raw_output,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return trace


def _extract_qea_scores(qea_output: Any) -> dict | None:
    """Pull the six headline QEA signals into a flat dict for analytics.

    Returns ``None`` if QEA never ran or its result is missing.
    """
    if qea_output is None:
        return None
    res = getattr(qea_output, "result", None)
    if res is None:
        return None
    try:
        return {
            "evidence_sufficiency": _safe_get_score(res.evidence_sufficiency),
            "rca_cmp_consistency": _safe_get_score(res.rca_cmp_consistency),
            "countermeasure_quality": _safe_get_score(res.countermeasure_quality),
            "verification_quality": _safe_get_score(res.verification_quality),
            "approval_readiness": _safe_get_score(res.approval_readiness),
            "hallucination_risk": _safe_get_field(
                res.hallucination_risk, "risk_level"
            ),
            "overall_judgement": getattr(res, "overall_judgement", None),
        }
    except Exception:
        return None


def _safe_get_score(section: Any) -> Any:
    return _safe_get_field(section, "score")


def _safe_get_field(section: Any, key: str) -> Any:
    if isinstance(section, dict):
        return section.get(key)
    return None
