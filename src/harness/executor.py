from collections.abc import Callable

from src.harness.schema import AgentInput, AgentOutput
from src.harness.validator import (
    validate_agent_input,
    validate_agent_output,
    ValidationError,
)


AgentFunction = Callable[[AgentInput], AgentOutput]


def execute_agent(
    agent_input: AgentInput,
    agent_function: AgentFunction,
) -> AgentOutput:
    """
    Execute an agent through the Harness.

    Responsibilities:
    - validate input
    - run agent
    - validate output
    - handle validation errors
    """

    try:
        validate_agent_input(agent_input)

        agent_output = agent_function(agent_input)

        validate_agent_output(agent_output)

        return agent_output

    except ValidationError as error:
        return AgentOutput(
            task_id=agent_input.task_id if hasattr(agent_input, "task_id") else "unknown",
            agent_name="harness",
            status="failure",
            result=None,
            confidence=0.0,
            errors=[str(error)],
            notes=["Execution stopped by Harness validation."],
        )