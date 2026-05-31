import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.quality_issue_analysis_agent import (
    QualityIssueAnalysisAgent,
    QualityIssueInput,
)
from src.agents.quality_issue_analysis_agent.quality_issue_analysis_agent import (
    analysis_to_dict,
)
from src.llm.base import LLMProvider


# -------------------------
# Built-in quality issue cases
# -------------------------
QUALITY_ISSUE_CASES: dict[str, QualityIssueInput] = {
    "screw_fastening": QualityIssueInput(
        case_id="quality-case-screw_fastening",
        process_name="screw_fastening",
        product_or_part="aluminum housing M4 screw joint",
        defect_mode="loose torque on assembled units",
        observed_symptoms=[
            "post-assembly torque audit shows 12% units below spec",
            "intermittent rattling reported in field returns",
        ],
        process_conditions={
            "tool": "DC electric driver",
            "target_torque_Nm": 1.2,
            "tolerance_Nm": 0.15,
            "speed_rpm": 600,
        },
        known_constraints=[
            "cannot change screw spec in this build",
            "cycle time must stay under 4 seconds",
        ],
        available_data=[
            "torque trace log per unit",
            "operator shift roster",
            "incoming screw lot certificates",
        ],
    ),
    "solder_wetting": QualityIssueInput(
        case_id="quality-case-solder_wetting",
        process_name="reflow_soldering",
        product_or_part="control PCB QFN48 package",
        defect_mode="poor solder wetting on QFN pads",
        observed_symptoms=[
            "AOI flags head-in-pillow on 8% of units",
            "dewetting visible on thermal pad after reflow",
        ],
        process_conditions={
            "paste": "SAC305 Type 4",
            "peak_temperature_c": 245,
            "time_above_liquidus_s": 60,
            "nitrogen_atmosphere": False,
        },
        known_constraints=[
            "oven profile change requires line revalidation",
            "paste supplier fixed for 3 months",
        ],
        available_data=[
            "reflow profile log",
            "paste inspection (SPI) data",
            "pad surface finish (ENIG) certificate",
        ],
    ),
    "dimensional_variation": QualityIssueInput(
        case_id="quality-case-dimensional_variation",
        process_name="cnc_turning",
        product_or_part="steel shaft outer diameter 12 h7",
        defect_mode="diameter variation exceeding tolerance",
        observed_symptoms=[
            "SPC chart shows Cpk drop from 1.6 to 0.9 over 3 days",
            "out-of-tolerance parts cluster at end of shift",
        ],
        process_conditions={
            "machine": "CNC lathe L-220",
            "tool": "CNMG carbide insert",
            "cutting_speed_mpm": 180,
            "feed_mm_rev": 0.12,
            "coolant": "soluble oil",
        },
        known_constraints=[
            "tool change interval fixed at 200 parts",
            "no spindle warm-up allowed during shift change",
        ],
        available_data=[
            "in-process gauge measurements",
            "tool wear log",
            "ambient temperature log",
        ],
    ),
}


def get_case(case_name: str) -> QualityIssueInput:
    if case_name not in QUALITY_ISSUE_CASES:
        raise KeyError(f"Unknown quality issue case: {case_name}")
    return QUALITY_ISSUE_CASES[case_name]


def run_case(
    provider_name: str,
    case_name: str,
    provider: LLMProvider | None = None,
) -> dict:
    issue_input = get_case(case_name)
    agent = QualityIssueAnalysisAgent(
        provider=provider,
        provider_name=provider_name,
    )

    started_at = perf_counter()
    agent_output = agent.run(issue_input)
    latency_sec = perf_counter() - started_at

    result_dict = analysis_to_dict(agent_output.result)

    return {
        "provider": provider_name,
        "model": getattr(agent.provider, "model", "unknown"),
        "case_id": issue_input.case_id,
        "agent_name": agent_output.agent_name,
        "status": agent_output.status,
        "needs_review": agent_output.needs_review,
        "latency_sec": round(latency_sec, 6),
        "parse_success": agent_output.parse_success,
        "schema_valid": agent_output.schema_valid,
        "error_type": agent_output.error_type,
        "cause_count": len(result_dict.get("suspected_causes", [])) if result_dict else 0,
        "error_count": len(agent_output.errors),
        "result": result_dict,
        "raw_output": agent_output.raw_output,
        "errors": agent_output.errors,
        "notes": agent_output.notes,
    }


def append_run_log(run_output: dict, log_path: Path, output_path: Path | None) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "case_id": run_output["case_id"],
        "provider": run_output["provider"],
        "model": run_output.get("model"),
        "agent_name": run_output.get("agent_name"),
        "status": run_output["status"],
        "needs_review": run_output.get("needs_review", False),
        "latency_sec": run_output.get("latency_sec"),
        "parse_success": run_output.get("parse_success", False),
        "schema_valid": run_output.get("schema_valid", False),
        "error_type": run_output.get("error_type"),
        "cause_count": run_output.get("cause_count", 0),
        "error_count": run_output.get("error_count", len(run_output["errors"])),
        "errors": run_output["errors"],
        "raw_output": run_output.get("raw_output", ""),
        "output_path": str(output_path) if output_path else None,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a predefined quality issue analysis case."
    )
    parser.add_argument("--provider", choices=("ollama", "fugu"), default="ollama")
    parser.add_argument(
        "--case", choices=tuple(QUALITY_ISSUE_CASES), required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("logs/quality_issue_runs.jsonl"),
        help="Append comparison metrics to this JSONL file.",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_const",
        const=None,
        dest="run_log",
        help="Disable comparison logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    started_at = perf_counter()
    try:
        output = run_case(args.provider, args.case)
    except ValueError as error:
        if args.run_log:
            append_run_log(
                {
                    "provider": args.provider,
                    "model": os.getenv("FUGU_MODEL") if args.provider == "fugu" else None,
                    "case_id": f"quality-case-{args.case}",
                    "agent_name": QualityIssueAnalysisAgent.agent_name,
                    "status": "failure",
                    "needs_review": True,
                    "latency_sec": round(perf_counter() - started_at, 6),
                    "parse_success": False,
                    "schema_valid": False,
                    "error_type": "configuration_error",
                    "cause_count": 0,
                    "error_count": 1,
                    "raw_output": "",
                    "errors": [str(error)],
                },
                args.run_log,
                args.output,
            )
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print(f"provider: {output['provider']}")
    print(f"case_id: {output['case_id']}")
    print(f"agent status: {output['status']}")
    print("parsed QualityIssueAnalysis:")
    print(json.dumps(output["result"], ensure_ascii=False, indent=2))
    print(f"errors: {json.dumps(output['errors'], ensure_ascii=False)}")
    print(f"notes: {json.dumps(output['notes'], ensure_ascii=False)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.run_log:
        append_run_log(output, args.run_log, args.output)

    return 0 if output["status"] != "failure" else 1


if __name__ == "__main__":
    raise SystemExit(main())
