from dataclasses import dataclass, field
from src.harness.schema import AgentOutput

EVALUATION_VERSION = "v0.2"

WEIGHTS = {
    "process_validity": 0.25,
    "quality_checkpoint_consistency": 0.30,
    "basis_validity": 0.20,
    "quantity_validity": 0.25,
}

PROCESS_TO_CHECKPOINT_RULES = {
    "machining": ["dimension"],
    "hole_processing": ["hole", "count"],
    "drilling": ["hole", "diameter"],
    "surface_finish": ["surface", "roughness"],
    "inspection": ["inspection"],
}

def count_findings(findings: list, level: str) -> int:
    return sum(1 for finding in findings if finding.get("level") == level)


def calculate_overall_score(scores: dict) -> float:
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += scores.get(key, 0.0) * weight
    return round(total, 3)

def build_evaluation_result(scores: dict, findings: list, schema_valid: bool = True) -> dict:
    overall = calculate_overall_score(scores)

    return {
        "version": EVALUATION_VERSION,
        "schema_valid": schema_valid,
        "scores": {
            **scores,
            "overall": overall,
        },
        "error_count": count_findings(findings, "error"),
        "warning_count": count_findings(findings, "warning"),
        "findings": findings,
    }

def evaluate_basis_validity(result) -> dict:
    findings = []
    processes = getattr(result, "manufacturing_processes", [])

    if not processes:
        findings.append({
            "level": "error",
            "category": "basis_validity",
            "message": "No manufacturing processes found. Cannot evaluate basis validity."
        })
        return {
            "score": 0.0,
            "findings": findings
        }

    missing_count = 0

    for process in processes:
        basis = getattr(process, "basis", None)
        process_name = getattr(process, "process_name", "unknown")

        if basis is None or str(basis).strip() == "":
            missing_count += 1
            findings.append({
                "level": "warning",
                "category": "basis_validity",
                "message": f"Basis is missing for process: {process_name}"
            })

    score = 1.0 - (missing_count / len(processes))

    return {
        "score": round(score, 3),
        "findings": findings
    }


def evaluate_quantity_validity(result) -> dict:
    findings = []
    processes = getattr(result, "manufacturing_processes", [])

    if not processes:
        findings.append({
            "level": "error",
            "category": "quantity_validity",
            "message": "No manufacturing processes found. Cannot evaluate quantity validity."
        })
        return {
            "score": 0.0,
            "findings": findings
        }

    invalid_count = 0
    total_count = 0

    for process in processes:
        quantity = getattr(process, "quantity", None)
        process_name = getattr(process, "process_name", "unknown")

        # quantity が必要な工程だけ評価したい場合はここで条件分岐も可能
        total_count += 1

        if quantity is None:
            invalid_count += 1
            findings.append({
                "level": "warning",
                "category": "quantity_validity",
                "message": f"Quantity is missing for process: {process_name}"
            })
        else:
            try:
                q = float(quantity)
                if q <= 0:
                    invalid_count += 1
                    findings.append({
                        "level": "warning",
                        "category": "quantity_validity",
                        "message": f"Invalid quantity (<=0) for process: {process_name}"
                    })
            except (ValueError, TypeError):
                invalid_count += 1
                findings.append({
                    "level": "warning",
                    "category": "quantity_validity",
                    "message": f"Non-numeric quantity for process: {process_name}"
                })

    score = 1.0 - (invalid_count / total_count)

    return {
        "score": round(score, 3),
        "findings": findings
    }


def evaluate_quality_checkpoint_consistency(result) -> dict:
    findings = []
    processes = getattr(result, "manufacturing_processes", [])
    checkpoints = getattr(result, "quality_checkpoints", [])

    if not processes:
        findings.append({
            "level": "error",
            "category": "quality_checkpoint_consistency",
            "message": "No manufacturing processes found. Cannot evaluate checkpoint consistency."
        })
        return {
            "score": 0.0,
            "findings": findings
        }

    if not checkpoints:
        findings.append({
            "level": "warning",
            "category": "quality_checkpoint_consistency",
            "message": "No quality checkpoints found."
        })
        return {
            "score": 0.0,
            "findings": findings
        }

    checkpoint_text = " ".join(
        str(getattr(checkpoint, "checkpoint", "")) + " " + 
        str(getattr(checkpoint, "reason", "")) + " " +
        str(getattr(checkpoint, "inspection_method", ""))
        for checkpoint in checkpoints
    ).lower()

    matched_count = 0
    target_count = 0

    for process in processes:
        process_name = str(getattr(process, "process_name", "")).lower()

        for rule_process, checkpoint_keywords in PROCESS_TO_CHECKPOINT_RULES.items():
            if rule_process in process_name:
                target_count += 1

                if any(keyword in checkpoint_text for keyword in checkpoint_keywords):
                    matched_count += 1
                else:
                    findings.append({
                        "level": "warning",
                        "category": "quality_checkpoint_consistency",
                        "message": f"No corresponding quality checkpoint found for process: {process_name}"
                    })

    if target_count == 0:
        findings.append({
            "level": "info",
            "category": "quality_checkpoint_consistency",
            "message": "No target process found for checkpoint consistency evaluation."
        })
        return {
            "score": 1.0,
            "findings": findings
        }

    score = matched_count / target_count

    return {
        "score": round(score, 3),
        "findings": findings
    }


def evaluate_process_validity(result) -> dict:
    findings = []
    processes = getattr(result, "manufacturing_processes", [])

    if not processes:
        findings.append({
            "level": "error",
            "category": "process_validity",
            "message": "No manufacturing processes found."
        })
        return {
            "score": 0.0,
            "findings": findings
        }

    invalid_count = 0

    for i, process in enumerate(processes):
        name = str(getattr(process, "process_name", "")).strip().lower()

        if not name or name in ("unknown", "n/a"):
            invalid_count += 1
            findings.append({
                "level": "warning",
                "category": "process_validity",
                "message": f"Invalid or missing process name at index {i}"
            })

    # スコア
    score = 1.0 - (invalid_count / len(processes))

    return {
        "score": round(score, 3),
        "findings": findings
    }


@dataclass
class EvaluationResult:
    task_id: str
    agent_name: str
    score: float
    passed: bool
    metrics: dict[str, float | int | bool]
    notes: list[str]
    version: str = EVALUATION_VERSION
    findings: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


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
    basis_result = {"score": 0.0, "findings": []}
    quantity_result = {"score": 0.0, "findings": []}
    consistency_result = {"score": 0.0, "findings": []}
    process_result = {"score": 0.0, "findings": []}

    if has_result:
        has_processes = len(result.manufacturing_processes) > 0
        has_quality_checkpoints = len(result.quality_checkpoints) > 0

        basis_result = evaluate_basis_validity(result)
        quantity_result = evaluate_quantity_validity(result)
        consistency_result = evaluate_quality_checkpoint_consistency(result)
        process_result = evaluate_process_validity(result)

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

    # v0.2（追加）
    score += basis_result["score"] * 0.1
    score += quantity_result["score"] * 0.1
    score += consistency_result["score"] * 0.1
    score += process_result["score"] * 0.1

    score = min(score, 1.0)

    all_findings = []
    all_findings.extend(process_result["findings"])
    all_findings.extend(consistency_result["findings"])
    all_findings.extend(basis_result["findings"])
    all_findings.extend(quantity_result["findings"])

    v2_scores = {
        "process_validity": process_result["score"],
        "quality_checkpoint_consistency": consistency_result["score"],
        "basis_validity": basis_result["score"],
        "quantity_validity": quantity_result["score"],
    }

    v2_result = build_evaluation_result(
        scores=v2_scores,
        findings=all_findings,
        schema_valid=schema_valid,
    )

    passed = (
        v2_result["scores"]["overall"] >= 0.8
        and v2_result["error_count"] == 0
        and v2_result["warning_count"] == 0
    )


    for finding in basis_result["findings"]:
        notes.append(finding["message"])

    for finding in quantity_result["findings"]:
        notes.append(finding["message"])

    for finding in consistency_result["findings"]:
        notes.append(finding["message"])

    for finding in process_result["findings"]:
        notes.append(finding["message"])

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
            "basis_validity": basis_result["score"],
            "quantity_validity": quantity_result["score"],
            "quality_checkpoint_consistency": consistency_result["score"],
            "process_validity": process_result["score"],
            "overall": v2_result["scores"]["overall"],
            "warning_count": v2_result["warning_count"],
        },
        notes=notes,
        version=v2_result["version"],
        findings=v2_result["findings"],
        scores=v2_result["scores"],
    )