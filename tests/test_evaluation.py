from src.harness.evaluation import evaluate_agent_output
from src.harness.schema import (
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
    QualityCheckpoint,
)


def test_evaluation_basic():
    processes = [
        ManufacturingProcess(
            process_name="machining",
            description="Machine φ10 hole",
            target_feature="φ10 hole",
            quantity=4,
            basis="Drawing indicates φ10 hole.",
        ),
        ManufacturingProcess(
            process_name="inspection",
            description="Inspect machined feature",
            target_feature="φ10 hole",
            quantity=1,
            basis="Inspection required after machining.",
        ),
    ]

    checkpoints = [
        QualityCheckpoint(
            checkpoint="dimension_check",
            reason="Confirm machined dimensions.",
            inspection_method="caliper",
        ),
    ]

    planning_result = PlanningResult(
        manufacturing_processes=processes,
        quality_checkpoints=checkpoints,
    )

    agent_output = AgentOutput(
        task_id="test_001",
        agent_name="mock_agent",
        status="success",
        result=planning_result,
        confidence=0.9,
        errors=[],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert 0.0 <= evaluation.score <= 1.0
    assert "basis_validity" in evaluation.metrics
    assert "quantity_validity" in evaluation.metrics
    assert "quality_checkpoint_consistency" in evaluation.metrics
    assert "process_validity" in evaluation.metrics


if __name__ == "__main__":
    test_evaluation_basic()
