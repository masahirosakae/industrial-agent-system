import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflows import (
    QualityWorkflowInput,
    run_quality_workflow,
    workflow_result_to_dict,
)


# -------------------------
# Built-in manufacturing-quality sample case
# -------------------------
SAMPLE_CASE = QualityWorkflowInput(
    case_id="QW-001",
    process_name="screw tightening",
    product_or_part="aluminum bracket assembly",
    defect_mode="intermittent screw loosening detected after vibration test",
    observed_symptoms=[
        "Loosening observed in 3 out of 20 samples after vibration test",
        "Defects concentrated in the night shift lot",
        "Tightening torque target is 1.15 N.m",
        "No confirmed tool calibration record for the affected lot",
    ],
    process_conditions={
        "tool": "DC electric driver",
        "target_torque_Nm": 1.15,
        "vibration_test_profile": "MIL-STD-810G",
        "lot_size": 20,
    },
    known_constraints=[
        "Tightening cycle time must remain under 3 seconds per fastener",
        "Cannot change screw specification in this build",
    ],
    available_data=[
        "Torque trace per fastener",
        "Vibration test pass/fail log",
        "Shift roster",
    ],
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 Quality Workflow (QIA -> RCA -> CMP -> QEA) end-to-end."
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "fugu"),
        default="ollama",
        help="LLM provider name to share across all four agents.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        help="Local Ollama model (ignored when --provider=fugu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path; if set, the full workflow result is written as JSON.",
    )
    return parser


def _print_summary(workflow_result, latency_sec: float) -> None:
    print(f"workflow:        {workflow_result.workflow_name}")
    print(f"case_id:         {workflow_result.case_id}")
    print(f"provider:        {workflow_result.provider_name}")
    print(f"status:          {workflow_result.status}")
    print(f"final_judgement: {workflow_result.final_judgement}")
    print(f"stopped_at:      {workflow_result.stopped_at}")
    print(f"latency_sec:     {round(latency_sec, 4)}")
    print(f"executed steps:  {list(workflow_result.steps)}")

    print()
    print("per-step status:")
    for name, output in workflow_result.steps.items():
        status = getattr(output, "status", "unknown")
        error_type = getattr(output, "error_type", None)
        print(f"  - {name}: status={status} error_type={error_type}")

    qea_output = workflow_result.steps.get("quality_evaluation_agent")
    qea_result = getattr(qea_output, "result", None) if qea_output else None
    if qea_result is not None:
        print()
        print("QEA scores:")
        print(
            f"  evidence_sufficiency:    {qea_result.evidence_sufficiency.get('score')}"
        )
        print(
            f"  rca_cmp_consistency:     {qea_result.rca_cmp_consistency.get('score')}"
        )
        print(
            f"  countermeasure_quality:  {qea_result.countermeasure_quality.get('score')}"
        )
        print(
            f"  verification_quality:    {qea_result.verification_quality.get('score')}"
        )
        print(
            f"  approval_readiness:      {qea_result.approval_readiness.get('score')}"
        )
        print(
            f"  hallucination_risk:      {qea_result.hallucination_risk.get('risk_level')}"
        )

    if workflow_result.review_reasons:
        print()
        print("review_reasons:")
        for reason in workflow_result.review_reasons:
            print(f"  - {reason}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    started_at = perf_counter()
    try:
        workflow_result = run_quality_workflow(
            SAMPLE_CASE,
            provider_name=args.provider,
            model=args.model,
        )
    except ValueError as error:
        # Most commonly: provider environment is misconfigured (e.g. Fugu
        # without FUGU_API_KEY). We do NOT want to leak Python tracebacks
        # to the operator, just a clean message.
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    latency_sec = perf_counter() - started_at
    _print_summary(workflow_result, latency_sec)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                workflow_result_to_dict(workflow_result),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print()
        print(f"saved: {args.output}")

    if workflow_result.status == "success":
        return 0
    if workflow_result.status == "needs_review":
        return 1
    return 2  # failure


if __name__ == "__main__":
    raise SystemExit(main())
