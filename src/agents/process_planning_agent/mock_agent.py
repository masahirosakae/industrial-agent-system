from src.harness.schema import (
    AgentInput,
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
    QualityCheckpoint,
)


def run_process_planning_agent(agent_input: AgentInput) -> AgentOutput:
    """
    Mock implementation of Process Planning Agent.
    This does NOT use LLM.
    It returns a fixed, schema-compliant response.
    """

    processes = [
        ManufacturingProcess(
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
    ]

    result = PlanningResult(
        manufacturing_processes=processes,
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