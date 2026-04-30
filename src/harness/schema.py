from dataclasses import dataclass, field
from typing import Any, Literal


InputType = Literal["drawing", "specification"]
FileType = Literal["pdf", "image", "text"]
AgentStatus = Literal["success", "failure", "needs_review"]


@dataclass
class Source:
    file_path: str
    file_type: FileType


@dataclass
class Metadata:
    part_name: str
    drawing_type: str
    created_at: str


@dataclass
class AgentInput:
    task_id: str
    input_type: InputType
    source: Source
    metadata: Metadata


@dataclass
class ManufacturingProcess:
    process_name: str
    description: str
    target_feature: str
    quantity: int | str
    basis: str


@dataclass
class WorkItem:
    work_name: str
    required_input: str
    expected_output: str


@dataclass
class QualityCheckpoint:
    checkpoint: str
    reason: str
    inspection_method: str


@dataclass
class PlanningResult:
    manufacturing_processes: list[ManufacturingProcess] = field(default_factory=list)
    work_items: list[WorkItem] = field(default_factory=list)
    quality_checkpoints: list[QualityCheckpoint] = field(default_factory=list)


@dataclass
class AgentOutput:
    task_id: str
    agent_name: str
    status: AgentStatus
    result: PlanningResult | None
    confidence: float
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)