from src.harness.schema import AgentInput, Source, Metadata
from src.harness.validator import (
    validate_agent_input,
    validate_agent_output,
)
from src.agents.process_planning_agent.mock_agent import (
    run_process_planning_agent,
)


def main():
    agent_input = AgentInput(
        task_id="test-001",
        input_type="drawing",
        source=Source(file_path="sample.pdf", file_type="pdf"),
        metadata=Metadata(
            part_name="test_part", drawing_type="plate", created_at="2026-01-01"
        ),
    )

    # Input validation
    validate_agent_input(agent_input)

    # Run agent
    output = run_process_planning_agent(agent_input)

    # Output validation
    validate_agent_output(output)

    print("SUCCESS")
    print(output)


if __name__ == "__main__":
    main()
