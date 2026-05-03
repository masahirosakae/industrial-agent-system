from dataclasses import dataclass, field
from typing import Literal


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
class QualityCheckpoint:
    checkpoint_type: str
    description: str
    inspection_method: str
    basis: list[str]


@dataclass
class ManufacturingProcess:
    process_id: str
    process_type: str
    process_name: str
    description: str
    target_feature: str
    quantity: int | float | None
    basis: list[str]
    quality_checkpoints: list[QualityCheckpoint]
    confidence: float
    needs_review: bool


@dataclass
class WorkItem:
    work_name: str
    required_input: str
    expected_output: str


@dataclass
class PlanningResult:
    manufacturing_processes: list[ManufacturingProcess]
    findings: list[dict] = field(default_factory=list)


@dataclass
class AgentOutput:
    task_id: str
    agent_name: str
    status: AgentStatus
    result: PlanningResult | None
    confidence: float
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)