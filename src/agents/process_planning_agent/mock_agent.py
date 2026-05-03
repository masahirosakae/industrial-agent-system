from src.harness.schema import (
    AgentInput,
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
    QualityCheckpoint,
)


def run_process_planning_agent(
    agent_input: AgentInput, mode: str = "valid"
) -> AgentOutput:

    process = ManufacturingProcess(
        process_id="P001",
        process_type="drilling",
        process_name="drilling",
        description="穴あけ加工",
        target_feature="holes",
        quantity=4,
        basis=["図面上の穴指示"],
        quality_checkpoints=[
            QualityCheckpoint(
                checkpoint_type="hole_diameter",
                description="穴径を確認する",
                inspection_method="measurement",
                basis=["図面上の穴径指示"],
            )
        ],
        confidence=0.8,
        needs_review=False,
    )

    if mode == "missing_basis":
        process.basis = []

    elif mode == "missing_quantity":
        process.quantity = None

    elif mode == "invalid_process_type":
        process.process_type = "invalid_type"

    elif mode == "checkpoint_mismatch":
        process.process_type = "drilling"
        process.quality_checkpoints[0].checkpoint_type = "thread_check"
        process.quality_checkpoints[0].description = "ねじ確認"

    result = PlanningResult(
        manufacturing_processes=[process],
        findings=[],
    )

    return AgentOutput(
        task_id=agent_input.task_id,
        agent_name="process_planning_agent",
        status="success",
        result=result,
        confidence=0.8,
        errors=[],
        notes=[],
    )
