import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)
from src.harness.executor import execute_agent
from src.harness.schema import AgentInput, Metadata, Source
from src.harness.validator import ValidationError, validate_agent_output
from src.llm.base import LLMProvider


CASE_FILE_PATHS = {
    "drilling": "穴_M8_4ヶ所.txt",
    "milling": "フライス_面加工_外形.txt",
    "unknown": "unknown_input_no_manufacturing_feature.txt",
}


def create_case(case_name: str) -> AgentInput:
    return AgentInput(
        task_id=f"planning-case-{case_name}",
        input_type="specification",
        source=Source(
            file_path=CASE_FILE_PATHS[case_name],
            file_type="text",
        ),
        metadata=Metadata(
            part_name="planning_case_sample",
            drawing_type="unknown",
            created_at="2026-05-31",
        ),
    )


def run_case(
    provider_name: str,
    case_name: str,
    provider: LLMProvider | None = None,
) -> dict:
    agent_input = create_case(case_name)
    agent = LocalLLMProcessPlanningAgent(
        provider=provider,
        provider_name=provider_name,
    )
    started_at = perf_counter()
    agent_output = execute_agent(
        agent_input=agent_input,
        agent_function=agent.run,
    )
    latency_sec = perf_counter() - started_at
    processes = (
        agent_output.result.manufacturing_processes if agent_output.result else []
    )

    try:
        validate_agent_output(agent_output)
        schema_valid = True
    except ValidationError:
        schema_valid = False

    parse_success = agent_output.result is not None and not any(
        finding.get("category") == "parse_error"
        for finding in agent_output.result.findings
    )

    return {
        "provider": provider_name,
        "model": getattr(agent.provider, "model", "unknown"),
        "case_id": agent_input.task_id,
        "agent_name": agent_output.agent_name,
        "status": agent_output.status,
        "latency_sec": round(latency_sec, 6),
        "parse_success": parse_success,
        "schema_valid": schema_valid,
        "process_count": len(processes),
        "needs_review": (
            agent_output.status == "needs_review"
            or any(process.needs_review for process in processes)
        ),
        "error_count": len(agent_output.errors),
        "result": asdict(agent_output.result) if agent_output.result else None,
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
        "latency_sec": run_output.get("latency_sec"),
        "parse_success": run_output.get("parse_success", False),
        "schema_valid": run_output.get("schema_valid", False),
        "process_count": run_output.get("process_count", 0),
        "needs_review": run_output.get("needs_review", False),
        "error_count": run_output.get("error_count", len(run_output["errors"])),
        "errors": run_output["errors"],
        "output_path": str(output_path) if output_path else None,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a predefined planning case.")
    parser.add_argument("--provider", choices=("ollama", "fugu"), default="ollama")
    parser.add_argument("--case", choices=tuple(CASE_FILE_PATHS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("logs/planning_runs.jsonl"),
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
                    "case_id": f"planning-case-{args.case}",
                    "agent_name": LocalLLMProcessPlanningAgent.agent_name,
                    "status": "failure",
                    "latency_sec": round(perf_counter() - started_at, 6),
                    "parse_success": False,
                    "schema_valid": False,
                    "process_count": 0,
                    "needs_review": False,
                    "error_count": 1,
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
    print("parsed PlanningResult:")
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
