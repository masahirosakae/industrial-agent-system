import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_planning_case import CASE_FILE_PATHS, append_run_log, run_case
from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)


PROVIDERS = ("ollama", "fugu")
SUMMARY_COLUMNS = (
    "provider",
    "case_id",
    "status",
    "parse_success",
    "schema_valid",
    "process_count",
    "needs_review",
    "latency_sec",
)


def parse_csv_list(value: str, allowed: tuple[str, ...], option_name: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid_items = [item for item in items if item not in allowed]

    if not items:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty")
    if invalid_items:
        raise argparse.ArgumentTypeError(
            f"Unsupported {option_name}: {', '.join(invalid_items)}"
        )

    return items


def create_failure_output(provider_name: str, case_name: str, error: Exception) -> dict:
    return {
        "provider": provider_name,
        "model": (
            os.getenv("FUGU_MODEL")
            if provider_name == "fugu"
            else "qwen2.5:1.5b"
        ),
        "case_id": f"planning-case-{case_name}",
        "agent_name": LocalLLMProcessPlanningAgent.agent_name,
        "status": "failure",
        "latency_sec": 0.0,
        "parse_success": False,
        "schema_valid": False,
        "process_count": 0,
        "needs_review": False,
        "error_count": 1,
        "errors": [str(error)],
    }


def run_comparison(
    providers: list[str],
    cases: list[str],
    log_path: Path,
) -> list[dict]:
    outputs = []

    for provider_name in providers:
        for case_name in cases:
            started_at = perf_counter()
            try:
                output = run_case(provider_name, case_name)
            except Exception as error:
                output = create_failure_output(provider_name, case_name, error)
                output["latency_sec"] = round(perf_counter() - started_at, 6)

            append_run_log(output, log_path, output_path=None)
            outputs.append(output)

    return outputs


def print_summary(outputs: list[dict]) -> None:
    rows = [
        [str(output.get(column, "")) for column in SUMMARY_COLUMNS]
        for output in outputs
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(SUMMARY_COLUMNS)
    ]

    print(" | ".join(column.ljust(widths[index]) for index, column in enumerate(SUMMARY_COLUMNS)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare predefined planning cases across LLM providers."
    )
    parser.add_argument(
        "--providers",
        type=lambda value: parse_csv_list(value, PROVIDERS, "providers"),
        default=["ollama"],
    )
    parser.add_argument(
        "--cases",
        type=lambda value: parse_csv_list(value, tuple(CASE_FILE_PATHS), "cases"),
        default=list(CASE_FILE_PATHS),
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("logs/planning_runs.jsonl"),
        help="Append comparison metrics to this JSONL file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_comparison(args.providers, args.cases, args.run_log)
    print_summary(outputs)
    return 0 if all(output["status"] != "failure" for output in outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
