from dataclasses import dataclass

from src.harness.schema import AgentOutput


@dataclass
class EvaluationResult:
    task_id: str
    agent_name: str
    score: float
    passed: bool
    metrics: dict[str, float | int | bool]
    notes: list[str]


def evaluate_agent_output(agent_output: AgentOutput) -> EvaluationResult:
    """
    Evaluate AgentOutput with simple rule-based metrics.
    This is a minimal v0.1 evaluation function.
    """

    notes: list[str] = []

    schema_valid = agent_output.status in ("success", "failure", "needs_review")
    result = agent_output.result

    has_result = result is not None
    has_processes = False
    has_quality_checkpoints = False

    if result is not None:
        has_processes = len(result.manufacturing_processes) > 0
        has_quality_checkpoints = len(result.quality_checkpoints) > 0

    confidence = agent_output.confidence
    error_count = len(agent_output.errors)

    score = 0.0

    if schema_valid:
        score += 0.2
    else:
        notes.append("Invalid status value.")

    if has_result:
        score += 0.2
    else:
        notes.append("Result is missing.")

    if has_processes:
        score += 0.2
    else:
        notes.append("No manufacturing processes found.")

    if has_quality_checkpoints:
        score += 0.2
    else:
        notes.append("No quality checkpoints found.")

    if confidence >= 0.7:
        score += 0.2
    else:
        notes.append("Confidence is below threshold.")

    passed = score >= 0.8 and error_count == 0

    return EvaluationResult(
        task_id=agent_output.task_id,
        agent_name=agent_output.agent_name,
        score=round(score, 2),
        passed=passed,
        metrics={
            "schema_valid": schema_valid,
            "has_result": has_result,
            "has_processes": has_processes,
            "has_quality_checkpoints": has_quality_checkpoints,
            "confidence": confidence,
            "error_count": error_count,
        },
        notes=notes,
    )