from src.harness.schema import AgentInput, Source, Metadata
from src.harness.executor import execute_agent
from src.agents.process_planning_agent.mock_agent import (
    run_process_planning_agent,
)
from src.harness.evaluation import evaluate_agent_output


def main():
    agent_input = AgentInput(
        task_id="test-failure-001",
        input_type="drawing",
        source=Source(
            file_path="",  # intentionally invalid
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

    evaluation = evaluate_agent_output(output)

    print("=== OUTPUT ===")
    print(output)

    print("\n=== EVALUATION ===")
    print(evaluation)


if __name__ == "__main__":
    main()
