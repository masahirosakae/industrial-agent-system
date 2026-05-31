import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.agents.quality_issue_analysis_agent import (
    QualityIssueAgentOutput,
    QualityIssueAnalysis,
)
from src.llm.base import LLMProvider
from src.llm.factory import create_llm_provider


RootCauseStatus = Literal["success", "failure", "needs_review"]


# -------------------------
# Input / Output schema
# -------------------------
@dataclass
class RootCauseAnalysisInput:
    """Input for RootCauseAnalysisAgent.

    Consumes the output of QualityIssueAnalysisAgent. The previous analysis
    is passed in directly so that this agent never re-interprets raw
    process data and instead reasons strictly on top of issue analysis
    facts and hypotheses.
    """

    case_id: str
    quality_issue_analysis: QualityIssueAnalysis
    process_name: str = ""
    product_or_part: str = ""

    @classmethod
    def from_quality_issue_output(
        cls,
        quality_issue_output: QualityIssueAgentOutput,
        process_name: str = "",
        product_or_part: str = "",
    ) -> "RootCauseAnalysisInput":
        """Build an RCA input from a trusted QualityIssueAnalysisAgent output.

        Only accepts a QIA output that fully cleared its own validation
        pipeline. Any fallback / needs_review / parse_error /
        schema_validation_error state is rejected so that the RCA agent
        never reasons on top of an untrusted prior analysis.
        """
        if quality_issue_output.result is None:
            raise ValueError(
                "QualityIssueAgentOutput.result is required to build RootCauseAnalysisInput"
            )

        problems: list[str] = []
        if quality_issue_output.status != "success":
            problems.append(f"status={quality_issue_output.status!r}")
        if quality_issue_output.needs_review:
            problems.append("needs_review=True")
        if not quality_issue_output.parse_success:
            problems.append("parse_success=False")
        if not quality_issue_output.schema_valid:
            problems.append("schema_valid=False")
        if problems:
            raise ValueError(
                "QualityIssueAgentOutput is not a trusted analysis: "
                + ", ".join(problems)
            )

        return cls(
            case_id=quality_issue_output.case_id,
            quality_issue_analysis=quality_issue_output.result,
            process_name=process_name,
            product_or_part=product_or_part,
        )


@dataclass
class RootCauseAnalysis:
    problem_statement: str
    facts: list[dict] = field(default_factory=list)
    assumptions: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    five_why: list[dict] = field(default_factory=list)
    fta_tree: dict = field(
        default_factory=lambda: {"top_event": "", "branches": []}
    )
    missing_evidence: list[dict] = field(default_factory=list)


@dataclass
class RootCauseAgentOutput:
    case_id: str
    agent_name: str
    status: RootCauseStatus
    result: RootCauseAnalysis | None
    confidence: float
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    parse_success: bool = False
    schema_valid: bool = False
    error_type: str | None = None
    raw_output: str = ""


# Required top-level keys for downstream consumers / validators.
ROOT_CAUSE_OUTPUT_SCHEMA = (
    "problem_statement",
    "facts",
    "assumptions",
    "hypotheses",
    "five_why",
    "fta_tree",
    "missing_evidence",
)

CONFIDENCE_VALUES = {"high", "medium", "low"}
EVIDENCE_STATUS_VALUES = {"confirmed", "assumed", "missing"}
PRIORITY_VALUES = {"high", "medium", "low"}


class RootCauseParseError(ValueError):
    pass


class RootCauseSchemaError(ValueError):
    pass


# -------------------------
# Agent
# -------------------------
class RootCauseAnalysisAgent:
    agent_name = "root_cause_analysis_agent"
    system_prompt = (
        "You are an industrial root cause analysis agent. "
        "Return valid JSON only and never invent facts that are not present "
        "in the provided quality issue analysis."
    )

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        provider: LLMProvider | None = None,
        provider_name: str | None = None,
    ):
        self.model = model
        self.provider = provider or create_llm_provider(
            provider_name=provider_name,
            ollama_model=self.model,
        )

    def run(self, rca_input: RootCauseAnalysisInput) -> RootCauseAgentOutput:
        raw_output = ""
        try:
            prompt = self._build_prompt(rca_input)
            raw_output = self.provider.generate(
                prompt,
                system_prompt=self.system_prompt,
            ).text
        except Exception as e:
            return self._needs_review_output(
                rca_input=rca_input,
                error=e,
                error_type="provider_error",
                raw_output=raw_output,
                parse_success=False,
                schema_valid=False,
                note="LLM provider execution failed; returning conservative review result",
            )

        try:
            data = self._parse_json_object(raw_output)
        except Exception as e:
            return self._needs_review_output(
                rca_input=rca_input,
                error=e,
                error_type="parse_error",
                raw_output=raw_output,
                parse_success=False,
                schema_valid=False,
                note="LLM output could not be parsed; returning conservative review result",
            )

        try:
            self._validate_raw_analysis(data)
        except Exception as e:
            return self._needs_review_output(
                rca_input=rca_input,
                error=e,
                error_type="schema_validation_error",
                raw_output=raw_output,
                parse_success=True,
                schema_valid=False,
                note="LLM output failed schema validation; returning conservative review result",
            )

        result = self._analysis_from_valid_data(data)
        return RootCauseAgentOutput(
            case_id=rca_input.case_id,
            agent_name=self.agent_name,
            status="success",
            result=result,
            confidence=0.7,
            errors=[],
            notes=["Generated by LLM provider"],
            needs_review=False,
            review_reasons=[],
            parse_success=True,
            schema_valid=True,
            error_type=None,
            raw_output=raw_output,
        )

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, rca_input: RootCauseAnalysisInput) -> str:
        prior_analysis = asdict(rca_input.quality_issue_analysis)
        payload = {
            "case_id": rca_input.case_id,
            "process_name": rca_input.process_name,
            "product_or_part": rca_input.product_or_part,
            "quality_issue_analysis": prior_analysis,
        }

        return f"""
Perform a root cause analysis on top of the provided quality issue analysis and return a structured JSON object only.

# Input
{json.dumps(payload, ensure_ascii=False, indent=2)}

# Required reasoning
- Perform a 5 Why analysis with exactly 3 to 5 levels (fewer or more is invalid).
- Build an FTA-style decomposition with a single top event and category branches.
- Strictly separate Fact, Assumption, and Hypothesis.
- List missing evidence explicitly with purpose and priority.
- Assign a confidence to each hypothesis.

# Output rules
- Return JSON only. No prose, no markdown, no code fences.
- Do not assert facts that are not present in the input quality issue analysis.
- evidence_status must be one of: confirmed, assumed, missing.
- confidence must be one of: high, medium, low.
- priority must be one of: high, medium, low.

# Required JSON schema
{{
  "problem_statement": "non-empty string",
  "facts": [
    {{"fact": "non-empty string", "source": "non-empty string"}}
  ],
  "assumptions": [
    {{"assumption": "non-empty string", "reason": "non-empty string"}}
  ],
  "hypotheses": [
    {{
      "hypothesis": "non-empty string",
      "confidence": "high|medium|low",
      "supporting_facts": ["non-empty string"],
      "missing_evidence": ["string"]
    }}
  ],
  "five_why": [
    {{
      "level": 1,
      "why": "non-empty string",
      "answer": "non-empty string",
      "evidence_status": "confirmed|assumed|missing"
    }}
  ],
  "fta_tree": {{
    "top_event": "non-empty string",
    "branches": [
      {{"category": "non-empty string", "causes": ["non-empty string"]}}
    ]
  }},
  "missing_evidence": [
    {{
      "evidence": "non-empty string",
      "purpose": "non-empty string",
      "priority": "high|medium|low"
    }}
  ]
}}
""".strip()

    # -------------------------
    # JSON parsing (strict whole-response only)
    # -------------------------
    def _parse_json_object(self, text: str) -> dict:
        """Parse the entire LLM response as a single JSON object.

        No prose/markdown tolerance, no code fence stripping, no
        ``{`` / ``}`` substring extraction. Any extra characters before
        or after the JSON object cause ``RootCauseParseError``.
        """
        if not isinstance(text, str):
            raise RootCauseParseError("LLM response must be a string")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RootCauseParseError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise RootCauseParseError("LLM response is not a JSON object")
        return data

    # -------------------------
    # Schema validation
    # -------------------------
    def _validate_raw_analysis(self, data: dict) -> None:
        errors: list[str] = []

        missing_keys = [key for key in ROOT_CAUSE_OUTPUT_SCHEMA if key not in data]
        if missing_keys:
            errors.append(f"missing required keys: {missing_keys}")

        self._validate_non_empty_string(data, "problem_statement", errors)
        self._validate_facts(data.get("facts"), errors)
        self._validate_assumptions(data.get("assumptions"), errors)
        self._validate_hypotheses(data.get("hypotheses"), errors)
        self._validate_five_why(data.get("five_why"), errors)
        self._validate_fta_tree(data.get("fta_tree"), errors)
        self._validate_missing_evidence(data.get("missing_evidence"), errors)

        if errors:
            raise RootCauseSchemaError("; ".join(errors))

    @staticmethod
    def _validate_non_empty_string(data: dict, key: str, errors: list[str]) -> None:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")

    @staticmethod
    def _validate_object_list(
        value,
        list_key: str,
        required_fields: tuple[str, ...],
        errors: list[str],
        require_non_empty: bool,
    ) -> None:
        if not isinstance(value, list):
            errors.append(f"{list_key} must be a list")
            return
        if require_non_empty and not value:
            errors.append(f"{list_key} must not be empty")
            return
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{list_key}[{index}] must be an object")
                continue
            for required_field in required_fields:
                item_value = item.get(required_field)
                if not isinstance(item_value, str) or not item_value.strip():
                    errors.append(
                        f"{list_key}[{index}].{required_field} must be a non-empty string"
                    )

    @classmethod
    def _validate_facts(cls, value, errors: list[str]) -> None:
        cls._validate_object_list(
            value, "facts", ("fact", "source"), errors, require_non_empty=False
        )

    @classmethod
    def _validate_assumptions(cls, value, errors: list[str]) -> None:
        cls._validate_object_list(
            value,
            "assumptions",
            ("assumption", "reason"),
            errors,
            require_non_empty=False,
        )

    @staticmethod
    def _validate_hypotheses(value, errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("hypotheses must be a list")
            return
        if not value:
            errors.append("hypotheses must not be empty")
            return
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"hypotheses[{index}] must be an object")
                continue
            hypothesis = item.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                errors.append(
                    f"hypotheses[{index}].hypothesis must be a non-empty string"
                )
            if item.get("confidence") not in CONFIDENCE_VALUES:
                errors.append(
                    f"hypotheses[{index}].confidence must be one of {sorted(CONFIDENCE_VALUES)}"
                )
            supporting = item.get("supporting_facts")
            if not isinstance(supporting, list):
                errors.append(
                    f"hypotheses[{index}].supporting_facts must be a list"
                )
            elif any(
                not isinstance(entry, str) or not entry.strip() for entry in supporting
            ):
                errors.append(
                    f"hypotheses[{index}].supporting_facts must contain non-empty strings"
                )
            missing = item.get("missing_evidence")
            if not isinstance(missing, list) or any(
                not isinstance(entry, str) for entry in missing
            ):
                errors.append(
                    f"hypotheses[{index}].missing_evidence must be a list of strings"
                )

    @staticmethod
    def _validate_five_why(value, errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("five_why must be a list")
            return
        if not value:
            errors.append("five_why must not be empty")
            return
        if len(value) < 3 or len(value) > 5:
            errors.append("five_why must have between 3 and 5 entries")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"five_why[{index}] must be an object")
                continue
            level = item.get("level")
            if not isinstance(level, int) or isinstance(level, bool) or level < 1:
                errors.append(
                    f"five_why[{index}].level must be a positive integer"
                )
            for key in ("why", "answer"):
                value_str = item.get(key)
                if not isinstance(value_str, str) or not value_str.strip():
                    errors.append(
                        f"five_why[{index}].{key} must be a non-empty string"
                    )
            if item.get("evidence_status") not in EVIDENCE_STATUS_VALUES:
                errors.append(
                    f"five_why[{index}].evidence_status must be one of {sorted(EVIDENCE_STATUS_VALUES)}"
                )

        # The per-item check above already rejects non-int / bool / non-positive
        # levels. In addition, require the sequence of levels to be exactly
        # ``[1, 2, ..., N]`` so the CountermeasurePlanningAgent can rely on
        # ordinal 5-Why reasoning (no duplicates, gaps, reverse order, 0-start,
        # or out-of-range values).
        actual_levels = [
            item.get("level") if isinstance(item, dict) else None
            for item in value
        ]
        expected_levels = list(range(1, len(value) + 1))
        if actual_levels != expected_levels:
            errors.append(
                "five_why levels must be a 1..N sequence; "
                f"got {actual_levels}, expected {expected_levels}"
            )

    @staticmethod
    def _validate_fta_tree(value, errors: list[str]) -> None:
        if not isinstance(value, dict):
            errors.append("fta_tree must be an object")
            return
        top_event = value.get("top_event")
        if not isinstance(top_event, str) or not top_event.strip():
            errors.append("fta_tree.top_event must be a non-empty string")
        branches = value.get("branches")
        if not isinstance(branches, list):
            errors.append("fta_tree.branches must be a list")
            return
        if not branches:
            errors.append("fta_tree.branches must not be empty")
            return
        for index, branch in enumerate(branches):
            if not isinstance(branch, dict):
                errors.append(f"fta_tree.branches[{index}] must be an object")
                continue
            category = branch.get("category")
            if not isinstance(category, str) or not category.strip():
                errors.append(
                    f"fta_tree.branches[{index}].category must be a non-empty string"
                )
            causes = branch.get("causes")
            if not isinstance(causes, list) or not causes:
                errors.append(
                    f"fta_tree.branches[{index}].causes must be a non-empty list"
                )
            elif any(
                not isinstance(cause, str) or not cause.strip() for cause in causes
            ):
                errors.append(
                    f"fta_tree.branches[{index}].causes must contain non-empty strings"
                )

    @staticmethod
    def _validate_missing_evidence(value, errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("missing_evidence must be a list")
            return
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"missing_evidence[{index}] must be an object")
                continue
            for key in ("evidence", "purpose"):
                value_str = item.get(key)
                if not isinstance(value_str, str) or not value_str.strip():
                    errors.append(
                        f"missing_evidence[{index}].{key} must be a non-empty string"
                    )
            if item.get("priority") not in PRIORITY_VALUES:
                errors.append(
                    f"missing_evidence[{index}].priority must be one of {sorted(PRIORITY_VALUES)}"
                )

    # -------------------------
    # Build dataclass / fallback
    # -------------------------
    @staticmethod
    def _analysis_from_valid_data(data: dict) -> RootCauseAnalysis:
        return RootCauseAnalysis(
            problem_statement=data["problem_statement"],
            facts=data["facts"],
            assumptions=data["assumptions"],
            hypotheses=data["hypotheses"],
            five_why=data["five_why"],
            fta_tree=data["fta_tree"],
            missing_evidence=data["missing_evidence"],
        )

    def _needs_review_output(
        self,
        rca_input: RootCauseAnalysisInput,
        error: Exception,
        error_type: str,
        raw_output: str,
        parse_success: bool,
        schema_valid: bool,
        note: str,
    ) -> RootCauseAgentOutput:
        reason = f"{error_type}: {error}"
        return RootCauseAgentOutput(
            case_id=rca_input.case_id,
            agent_name=self.agent_name,
            status="needs_review",
            result=self._fallback_result(),
            confidence=0.0,
            errors=[str(error)],
            notes=[note],
            needs_review=True,
            review_reasons=[reason],
            parse_success=parse_success,
            schema_valid=schema_valid,
            error_type=error_type,
            raw_output=raw_output,
        )

    @staticmethod
    def _fallback_result() -> RootCauseAnalysis:
        return RootCauseAnalysis(
            problem_statement="",
            facts=[],
            assumptions=[],
            hypotheses=[],
            five_why=[],
            fta_tree={"top_event": "", "branches": []},
            missing_evidence=[],
        )


def analysis_to_dict(analysis: RootCauseAnalysis | None) -> dict | None:
    return asdict(analysis) if analysis else None
