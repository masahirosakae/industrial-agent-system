import json
from datetime import datetime
from pathlib import Path

from src.harness.schema import AgentInput, AgentOutput


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log_execution(
    agent_input: AgentInput,
    agent_output: AgentOutput,
) -> None:
    """
    Log a single agent execution.
    """

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": agent_input.task_id,
        "agent_name": agent_output.agent_name,
        "status": agent_output.status,
        "input": serialize(agent_input),
        "output": serialize(agent_output),
    }

    log_file = LOG_DIR / f"{agent_input.task_id}.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def serialize(obj):
    """
    Convert dataclass to dict recursively.
    """
    if hasattr(obj, "__dict__"):
        return {
            key: serialize(value)
            for key, value in obj.__dict__.items()
        }
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    else:
        return obj