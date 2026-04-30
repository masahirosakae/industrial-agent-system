from src.harness.schema import AgentInput, Source, Metadata
from src.harness.executor import execute_agent
from src.agents.process_planning_agent.mock_agent import (
    run_process_planning_agent,
)


def main():
    agent_input = AgentInput(
        task_id="test-002",
        input_type="drawing",
        source=Source(
            file_path="sample.pdf",
            file_type="pdf",
        ),
        metadata=Metadata(
            part_name="test_part",
            drawing_type="plate",
            created_at="2026-01-01",
        ),
    )

    output = execute_agent(
        agent_input=agent_input,
        agent_function=run_process_planning_agent,
    )

    print(output)


if __name__ == "__main__":
    main()