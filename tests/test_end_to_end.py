from src.harness.schema import (
    AgentInput,
    Source,
    Metadata,
)
from src.harness.evaluation import evaluate_agent_output
from src.agents.process_planning_agent.mock_agent import run_process_planning_agent


def test_end_to_end_basic():

    # ===== ① 入力（仕様テキスト）=====
    agent_input = AgentInput(
        task_id="e2e_001",
        input_type="specification",
        source=Source(
            file_path="dummy_spec.txt",
            file_type="text",
        ),
        metadata=Metadata(
            part_name="Test Plate",
            drawing_type="plate",
            created_at="2026-01-01",
        ),
    )

    # ===== ② Agent実行 =====
    agent_output = run_process_planning_agent(agent_input)

    # ===== ③ evaluation =====
    evaluation = evaluate_agent_output(agent_output)

    print("\n=== Agent Output ===")
    print(agent_output)

    print("\n=== Evaluation ===")
    print(evaluation)

    # ===== ④ 最低限の検証 =====
    assert agent_output is not None
    assert evaluation is not None
    assert "overall" in evaluation.metrics

if __name__ == "__main__":
    test_end_to_end_basic()