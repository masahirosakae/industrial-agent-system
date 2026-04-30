from dataclasses import is_dataclass
from typing import Any

from src.harness.schema import AgentInput, AgentOutput


class ValidationError(Exception):
    """Harness validation error."""
    pass


def validate_agent_input(agent_input: AgentInput) -> None:
    """
    Validate AgentInput before agent execution.
    """

    if not is_dataclass(agent_input):
        raise ValidationError("agent_input must be a dataclass instance.")

    if not agent_input.task_id:
        raise ValidationError("task_id is required.")

    if agent_input.input_type not in ("drawing", "specification"):
        raise ValidationError("input_type must be 'drawing' or 'specification'.")

    if not agent_input.source.file_path:
        raise ValidationError("source.file_path is required.")

    if agent_input.source.file_type not in ("pdf", "image", "text"):
        raise ValidationError("source.file_type must be 'pdf', 'image', or 'text'.")

    if not agent_input.metadata.part_name:
        raise ValidationError("metadata.part_name is required.")

    if not agent_input.metadata.created_at:
        raise ValidationError("metadata.created_at is required.")


def validate_agent_output(agent_output: AgentOutput) -> None:
    """
    Validate AgentOutput after agent execution.
    """

    if not is_dataclass(agent_output):
        raise ValidationError("agent_output must be a dataclass instance.")

    if not agent_output.task_id:
        raise ValidationError("task_id is required.")

    if not agent_output.agent_name:
        raise ValidationError("agent_name is required.")

    if agent_output.status not in ("success", "failure", "needs_review"):
        raise ValidationError("status must be 'success', 'failure', or 'needs_review'.")

    if not 0.0 <= agent_output.confidence <= 1.0:
        raise ValidationError("confidence must be between 0.0 and 1.0.")

    if agent_output.status == "success" and agent_output.errors:
        raise ValidationError("successful output should not contain errors.")

    if agent_output.status == "failure" and not agent_output.errors:
        raise ValidationError("failure output must contain at least one error.")

    if agent_output.status == "needs_review" and not agent_output.notes:
        raise ValidationError("needs_review output should contain review notes.")


def validate_no_guessing_text(text: str) -> None:
    """
    Detect risky expressions that may indicate unsupported guessing.
    This is a simple placeholder rule for v0.1.
    """

    risky_patterns = [
        "おそらく",
        "たぶん",
        "推測",
        "推定",
        "と思われる",
        "likely",
        "probably",
        "maybe",
    ]

    for pattern in risky_patterns:
        if pattern in text:
            raise ValidationError(f"Potential guessing detected: {pattern}")


def validate_required_keys(data: dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate required keys in a raw dictionary.
    Useful before converting JSON into dataclasses.
    """

    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        raise ValidationError(f"Missing required keys: {missing_keys}")