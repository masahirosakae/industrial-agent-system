from src.harness.schema import (
    AgentInput,
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
    WorkItem,
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
            process_name="drilling",
            description="穴あけ加工",
            target_feature="holes",
            quantity=4,
            basis="図面上の穴指示"
        )
    ]

    work_items = [
        WorkItem(
            work_name="setup",
            required_input="tooling",
            expected_output="ready state"
        )
    ]

    quality_checkpoints = [
        QualityCheckpoint(
            checkpoint="hole_diameter",
            reason="重要寸法",
            inspection_method="measurement"
        )
    ]

    result = PlanningResult(
        manufacturing_processes=processes,
        work_items=work_items,
        quality_checkpoints=quality_checkpoints
    )

    return AgentOutput(
        task_id=agent_input.task_id,
        agent_name="process_planning_agent",
        status="success",
        result=result,
        confidence=0.8,
        errors=[],
        notes=[]
    )