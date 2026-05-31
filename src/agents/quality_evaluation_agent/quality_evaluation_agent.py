import json
from dataclasses import asdict, dataclass, field
from typing import Literal

# NOTE: This agent depends only on the abstract LLMProvider contract and the
# schema dataclasses exposed by the previous Quality Workflow stages. Concrete
# provider implementations must never be imported here; the provider-agnostic
# test guards this contract.
from src.agents.countermeasure_planning_agent import (
    CountermeasurePlan,
    CountermeasurePlanningAgentOutput,
)
from src.agents.root_cause_analysis_agent import (
    RootCauseAgentOutput,
    RootCauseAnalysis,
)
from src.llm.base import LLMProvider
from src.llm.factory import create_llm_provider


QualityEvaluationStatus = Literal["success", "failure", "needs_review"]


# -------------------------
# Input / Output schema
# -------------------------
@dataclass
class QualityEvaluationInput:
    """Input for QualityEvaluationAgent.

    Consumes the outputs of RootCauseAnalysisAgent AND
    CountermeasurePlanningAgent. Mirrors the upstream contracts: both prior
    analyses are passed in directly so this agent never re-derives facts and
    only evaluates already-validated reasoning.
    """

    case_id: str
    root_cause_analysis: RootCauseAnalysis
    countermeasure_plan: CountermeasurePlan
    process_name: str = ""
    product_or_part: str = ""

    @classmethod
    def from_previous_outputs(
        cls,
        root_cause_output: RootCauseAgentOutput,
        countermeasure_planning_output: CountermeasurePlanningAgentOutput,
        process_name: str = "",
        product_or_part: str = "",
    ) -> "QualityEvaluationInput":
        """Build a QEA input from trusted RCA + CMP outputs.

        Only accepts upstream outputs that fully cleared their own validation
        pipelines. Any fallback / needs_review / parse_error /
        schema_validation_error state on either side is rejected so the
        quality gate never reviews an untrusted intermediate analysis.
        """
        if root_cause_output.result is None:
            raise ValueError(
                "RootCauseAgentOutput.result is required to build QualityEvaluationInput"
            )
        if countermeasure_planning_output.result is None:
            raise ValueError(
                "CountermeasurePlanningAgentOutput.result is required to build QualityEvaluationInput"
            )

        rca_problems = cls._gate_problems(root_cause_output, "RootCauseAgentOutput")
        cmp_problems = cls._gate_problems(
            countermeasure_planning_output, "CountermeasurePlanningAgentOutput"
        )
        problems = rca_problems + cmp_problems
        if problems:
            raise ValueError(
                "Upstream Quality Workflow output is not a trusted analysis: "
                + "; ".join(problems)
            )

        return cls(
            case_id=root_cause_output.case_id,
            root_cause_analysis=root_cause_output.result,
            countermeasure_plan=countermeasure_planning_output.result,
            process_name=process_name,
            product_or_part=product_or_part,
        )

    @staticmethod
    def _gate_problems(upstream_output, label: str) -> list[str]:
        problems: list[str] = []
        if upstream_output.status != "success":
            problems.append(f"{label} status={upstream_output.status!r}")
        if upstream_output.needs_review:
            problems.append(f"{label} needs_review=True")
        if not upstream_output.parse_success:
            problems.append(f"{label} parse_success=False")
        if not upstream_output.schema_valid:
            problems.append(f"{label} schema_valid=False")
        return problems


@dataclass
class QualityEvaluationResult:
    evidence_sufficiency: dict = field(
        default_factory=lambda: {
            "score": 0.0,
            "assessment": "",
            "insufficient_items": [],
        }
    )
    hallucination_risk: dict = field(
        default_factory=lambda: {
            "risk_level": "high",
            "suspected_items": [],
        }
    )
    rca_cmp_consistency: dict = field(
        default_factory=lambda: {
            "score": 0.0,
            "assessment": "",
            "inconsistencies": [],
        }
    )
    countermeasure_quality: dict = field(
        default_factory=lambda: {
            "score": 0.0,
            "assessment": "",
            "weaknesses": [],
        }
    )
    verification_quality: dict = field(
        default_factory=lambda: {
            "score": 0.0,
            "assessment": "",
            "missing_verifications": [],
        }
    )
    approval_readiness: dict = field(
        default_factory=lambda: {
            "score": 0.0,
            "assessment": "",
        }
    )
    overall_judgement: str = "needs_review"
    review_reasons: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)


@dataclass
class QualityEvaluationAgentOutput:
    case_id: str
    status: QualityEvaluationStatus
    result: QualityEvaluationResult | None
    confidence: float
    agent_name: str = "quality_evaluation_agent"
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    parse_success: bool = False
    schema_valid: bool = False
    error_type: str | None = None
    raw_output: str = ""


# Required top-level keys for downstream consumers / validators.
QUALITY_EVALUATION_OUTPUT_SCHEMA = (
    "evidence_sufficiency",
    "hallucination_risk",
    "rca_cmp_consistency",
    "countermeasure_quality",
    "verification_quality",
    "approval_readiness",
    "overall_judgement",
    "review_reasons",
    "recommended_next_steps",
)

RISK_LEVEL_VALUES = {"low", "medium", "high"}
OVERALL_JUDGEMENT_VALUES = {"accepted", "needs_review"}
SCORE_THRESHOLD = 0.5


class QualityEvaluationParseError(ValueError):
    pass


class QualityEvaluationSchemaError(ValueError):
    pass


# -------------------------
# Agent
# -------------------------
class QualityEvaluationAgent:
    agent_name = "quality_evaluation_agent"
    system_prompt = (
        "You are an industrial quality evaluation agent acting as the final "
        "quality gate. Return valid JSON only. Never invent facts, equipment, "
        "processes, causes, or constraints that are not present in the "
        "provided root cause analysis or countermeasure plan."
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

    def run(
        self, eval_input: QualityEvaluationInput
    ) -> QualityEvaluationAgentOutput:
        raw_output = ""
        try:
            prompt = self._build_prompt(eval_input)
            raw_output = self.provider.generate(
                prompt,
                system_prompt=self.system_prompt,
            ).text
        except Exception as e:
            return self._needs_review_output(
                eval_input=eval_input,
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
                eval_input=eval_input,
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
                eval_input=eval_input,
                error=e,
                error_type="schema_validation_error",
                raw_output=raw_output,
                parse_success=True,
                schema_valid=False,
                note="LLM output failed schema validation; returning conservative review result",
            )

        # Schema is valid. Compute policy-override review reasons and decide
        # the final overall_judgement.
        model_review_reasons = list(data.get("review_reasons") or [])
        override_reasons = self._policy_override_reasons(data)
        merged_review_reasons = self._merge_strings(
            model_review_reasons, override_reasons
        )

        model_judgement = data["overall_judgement"]
        final_judgement = (
            "needs_review"
            if (model_judgement == "needs_review" or override_reasons)
            else "accepted"
        )

        result = QualityEvaluationResult(
            evidence_sufficiency=data["evidence_sufficiency"],
            hallucination_risk=data["hallucination_risk"],
            rca_cmp_consistency=data["rca_cmp_consistency"],
            countermeasure_quality=data["countermeasure_quality"],
            verification_quality=data["verification_quality"],
            approval_readiness=data["approval_readiness"],
            overall_judgement=final_judgement,
            review_reasons=(
                merged_review_reasons
                if final_judgement == "needs_review"
                else []
            ),
            recommended_next_steps=list(data.get("recommended_next_steps") or []),
        )

        if final_judgement == "needs_review":
            # Contract: every needs_review return path must carry at least
            # one human-readable review_reasons entry. If the LLM declared
            # overall_judgement=needs_review without populating review_reasons
            # and no policy override fired, inject a default so downstream
            # tooling never sees ``needs_review=True`` paired with ``[]``.
            if not merged_review_reasons:
                merged_review_reasons = [
                    "LLM reported overall_judgement=needs_review without "
                    "populating review_reasons"
                ]
            return QualityEvaluationAgentOutput(
                case_id=eval_input.case_id,
                status="needs_review",
                result=QualityEvaluationResult(
                    evidence_sufficiency=result.evidence_sufficiency,
                    hallucination_risk=result.hallucination_risk,
                    rca_cmp_consistency=result.rca_cmp_consistency,
                    countermeasure_quality=result.countermeasure_quality,
                    verification_quality=result.verification_quality,
                    approval_readiness=result.approval_readiness,
                    overall_judgement=result.overall_judgement,
                    review_reasons=list(merged_review_reasons),
                    recommended_next_steps=result.recommended_next_steps,
                ),
                confidence=0.5,
                errors=[],
                notes=[
                    "Schema is valid but policy review is required; "
                    "see review_reasons."
                ],
                needs_review=True,
                review_reasons=merged_review_reasons,
                parse_success=True,
                schema_valid=True,
                error_type="policy_review_required",
                raw_output=raw_output,
            )

        return QualityEvaluationAgentOutput(
            case_id=eval_input.case_id,
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

    @staticmethod
    def _merge_strings(*sources: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for source in sources:
            for item in source or []:
                if not isinstance(item, str):
                    continue
                if item in seen:
                    continue
                merged.append(item)
                seen.add(item)
        return merged

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, eval_input: QualityEvaluationInput) -> str:
        payload = {
            "case_id": eval_input.case_id,
            "process_name": eval_input.process_name,
            "product_or_part": eval_input.product_or_part,
            "root_cause_analysis": asdict(eval_input.root_cause_analysis),
            "countermeasure_plan": asdict(eval_input.countermeasure_plan),
        }

        return f"""
You are the final quality gate for the Phase 1 Quality Workflow. Evaluate
whether the provided root cause analysis and countermeasure plan are
trustworthy enough to present to a human engineer for approval, and return a
structured JSON object only.

# Input
{json.dumps(payload, ensure_ascii=False, indent=2)}

# Evaluation rules
- Strictly compare the countermeasure plan against the root cause analysis.
- Do not assert facts, equipment, processes, causes, or constraints that are
  not present in the input root cause analysis or countermeasure plan.
- If anything in the plan references entities that the analysis does not
  contain, treat it as a hallucination signal and populate
  hallucination_risk.suspected_items.
- If evidence is insufficient anywhere (RCA missing_evidence not covered,
  five_why with assumed / missing evidence_status, low-confidence hypotheses
  driving permanent actions), reflect that in evidence_sufficiency and
  set overall_judgement to "needs_review".
- If RCA hypotheses and CMP related_hypothesis or actions are inconsistent,
  populate rca_cmp_consistency.inconsistencies.
- If countermeasure quality is weak (no measurable success_criteria, missing
  containment, unrealistic implementation), populate
  countermeasure_quality.weaknesses.
- If verification_plan does not cover each hypothesis or missing_evidence,
  populate verification_quality.missing_verifications.
- approval_readiness must be the holistic score reflecting evidence
  sufficiency, hallucination risk, consistency, countermeasure quality,
  verification quality, residual risks, and any review_reasons.
- If hallucination_risk.risk_level is "high", overall_judgement MUST be
  "needs_review".
- recommended_next_steps must be concrete, actionable, and grounded in the
  input.

# Output rules
- Return JSON only. No prose, no markdown, no code fences.
- Each score must be a number between 0.0 and 1.0 (inclusive).
- hallucination_risk.risk_level must be one of: low, medium, high.
- overall_judgement must be one of: accepted, needs_review.
- review_reasons MUST be non-empty when overall_judgement is "needs_review".
- All list fields may be empty when no issues are detected.

# Required JSON schema
{{
  "evidence_sufficiency": {{
    "score": 0.0,
    "assessment": "non-empty string",
    "insufficient_items": ["string"]
  }},
  "hallucination_risk": {{
    "risk_level": "low|medium|high",
    "suspected_items": [
      {{"item": "non-empty string", "reason": "non-empty string"}}
    ]
  }},
  "rca_cmp_consistency": {{
    "score": 0.0,
    "assessment": "non-empty string",
    "inconsistencies": ["string"]
  }},
  "countermeasure_quality": {{
    "score": 0.0,
    "assessment": "non-empty string",
    "weaknesses": ["string"]
  }},
  "verification_quality": {{
    "score": 0.0,
    "assessment": "non-empty string",
    "missing_verifications": ["string"]
  }},
  "approval_readiness": {{
    "score": 0.0,
    "assessment": "non-empty string"
  }},
  "overall_judgement": "accepted|needs_review",
  "review_reasons": ["string"],
  "recommended_next_steps": ["string"]
}}
""".strip()

    # -------------------------
    # JSON parsing (strict whole-response only)
    # -------------------------
    def _parse_json_object(self, text: str) -> dict:
        """Parse the entire LLM response as a single JSON object.

        Same policy as :class:`QualityIssueAnalysisAgent`,
        :class:`RootCauseAnalysisAgent`, and
        :class:`CountermeasurePlanningAgent`: ``json.loads(text)`` only, no
        prose / markdown tolerance, no code fence stripping, no
        ``{`` / ``}`` substring extraction.
        """
        if not isinstance(text, str):
            raise QualityEvaluationParseError("LLM response must be a string")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise QualityEvaluationParseError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise QualityEvaluationParseError("LLM response is not a JSON object")
        return data

    # -------------------------
    # Schema validation
    # -------------------------
    def _validate_raw_analysis(self, data: dict) -> None:
        errors: list[str] = []

        missing_keys = [
            key for key in QUALITY_EVALUATION_OUTPUT_SCHEMA if key not in data
        ]
        if missing_keys:
            errors.append(f"missing required keys: {missing_keys}")

        self._validate_scored_section(
            data.get("evidence_sufficiency"),
            section_key="evidence_sufficiency",
            list_field="insufficient_items",
            list_item_validator=self._validate_string_list_items,
            errors=errors,
        )
        self._validate_hallucination_risk(data.get("hallucination_risk"), errors)
        self._validate_scored_section(
            data.get("rca_cmp_consistency"),
            section_key="rca_cmp_consistency",
            list_field="inconsistencies",
            list_item_validator=self._validate_string_list_items,
            errors=errors,
        )
        self._validate_scored_section(
            data.get("countermeasure_quality"),
            section_key="countermeasure_quality",
            list_field="weaknesses",
            list_item_validator=self._validate_string_list_items,
            errors=errors,
        )
        self._validate_scored_section(
            data.get("verification_quality"),
            section_key="verification_quality",
            list_field="missing_verifications",
            list_item_validator=self._validate_string_list_items,
            errors=errors,
        )
        self._validate_approval_readiness(data.get("approval_readiness"), errors)
        self._validate_overall_judgement(data.get("overall_judgement"), errors)
        self._validate_string_list_field(
            data.get("review_reasons"), key="review_reasons", errors=errors
        )
        self._validate_string_list_field(
            data.get("recommended_next_steps"),
            key="recommended_next_steps",
            errors=errors,
        )

        if errors:
            raise QualityEvaluationSchemaError("; ".join(errors))

    @staticmethod
    def _validate_score(value, key: str, errors: list[str]) -> None:
        if isinstance(value, bool):
            errors.append(f"{key} must be a number, got bool")
            return
        if not isinstance(value, (int, float)):
            errors.append(f"{key} must be a number")
            return
        if not (0.0 <= float(value) <= 1.0):
            errors.append(f"{key} must be between 0.0 and 1.0")

    @staticmethod
    def _validate_non_empty_string(value, key: str, errors: list[str]) -> None:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")

    @classmethod
    def _validate_string_list_field(
        cls, value, key: str, errors: list[str]
    ) -> None:
        if not isinstance(value, list):
            errors.append(f"{key} must be a list")
            return
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{key}[{index}] must be a non-empty string")

    @classmethod
    def _validate_string_list_items(
        cls, value, key: str, errors: list[str]
    ) -> None:
        cls._validate_string_list_field(value, key, errors)

    @classmethod
    def _validate_scored_section(
        cls,
        section,
        section_key: str,
        list_field: str,
        list_item_validator,
        errors: list[str],
    ) -> None:
        if not isinstance(section, dict):
            errors.append(f"{section_key} must be an object")
            return
        cls._validate_score(section.get("score"), f"{section_key}.score", errors)
        cls._validate_non_empty_string(
            section.get("assessment"), f"{section_key}.assessment", errors
        )
        list_item_validator(
            section.get(list_field), f"{section_key}.{list_field}", errors
        )

    @classmethod
    def _validate_hallucination_risk(cls, section, errors: list[str]) -> None:
        if not isinstance(section, dict):
            errors.append("hallucination_risk must be an object")
            return
        risk_level = section.get("risk_level")
        if risk_level not in RISK_LEVEL_VALUES:
            errors.append(
                f"hallucination_risk.risk_level must be one of {sorted(RISK_LEVEL_VALUES)}"
            )

        suspected_items = section.get("suspected_items")
        if not isinstance(suspected_items, list):
            errors.append("hallucination_risk.suspected_items must be a list")
            return
        for index, item in enumerate(suspected_items):
            if not isinstance(item, dict):
                errors.append(
                    f"hallucination_risk.suspected_items[{index}] must be an object"
                )
                continue
            for required_field in ("item", "reason"):
                cls._validate_non_empty_string(
                    item.get(required_field),
                    f"hallucination_risk.suspected_items[{index}].{required_field}",
                    errors,
                )

    @classmethod
    def _validate_approval_readiness(cls, section, errors: list[str]) -> None:
        if not isinstance(section, dict):
            errors.append("approval_readiness must be an object")
            return
        cls._validate_score(
            section.get("score"), "approval_readiness.score", errors
        )
        cls._validate_non_empty_string(
            section.get("assessment"), "approval_readiness.assessment", errors
        )

    @staticmethod
    def _validate_overall_judgement(value, errors: list[str]) -> None:
        if value not in OVERALL_JUDGEMENT_VALUES:
            errors.append(
                f"overall_judgement must be one of {sorted(OVERALL_JUDGEMENT_VALUES)}"
            )

    # -------------------------
    # Policy / safety override (runs only after schema is valid)
    # -------------------------
    def _policy_override_reasons(self, data: dict) -> list[str]:
        reasons: list[str] = []

        hallucination = data.get("hallucination_risk") or {}
        if hallucination.get("risk_level") == "high":
            reasons.append("hallucination_risk.risk_level is high")

        score_paths = (
            ("evidence_sufficiency", "score"),
            ("rca_cmp_consistency", "score"),
            ("countermeasure_quality", "score"),
            ("verification_quality", "score"),
            ("approval_readiness", "score"),
        )
        for parent, child in score_paths:
            score = (data.get(parent) or {}).get(child)
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and float(score) < SCORE_THRESHOLD
            ):
                reasons.append(
                    f"{parent}.{child} ({score}) is below threshold "
                    f"{SCORE_THRESHOLD}"
                )

        if data.get("review_reasons"):
            reasons.append("review_reasons is non-empty")

        non_empty_list_paths = (
            ("evidence_sufficiency", "insufficient_items"),
            ("hallucination_risk", "suspected_items"),
            ("rca_cmp_consistency", "inconsistencies"),
            ("countermeasure_quality", "weaknesses"),
            ("verification_quality", "missing_verifications"),
        )
        for parent, child in non_empty_list_paths:
            value = (data.get(parent) or {}).get(child)
            if isinstance(value, list) and value:
                reasons.append(f"{parent}.{child} is non-empty")

        return reasons

    # -------------------------
    # Fallback
    # -------------------------
    def _needs_review_output(
        self,
        eval_input: QualityEvaluationInput,
        error: Exception,
        error_type: str,
        raw_output: str,
        parse_success: bool,
        schema_valid: bool,
        note: str,
    ) -> QualityEvaluationAgentOutput:
        reason = f"{error_type}: {error}"
        return QualityEvaluationAgentOutput(
            case_id=eval_input.case_id,
            status="needs_review",
            result=self._fallback_result(reason),
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
    def _fallback_result(reason: str) -> QualityEvaluationResult:
        # Conservative empty evaluation envelope. Human review is the only
        # safe next action: we never claim "accepted" here, and we force
        # hallucination_risk to "high" so any downstream gating defaults to
        # the safe side when the model output was unavailable, malformed,
        # or failed schema validation.
        return QualityEvaluationResult(
            evidence_sufficiency={
                "score": 0.0,
                "assessment": "",
                "insufficient_items": [],
            },
            hallucination_risk={
                "risk_level": "high",
                "suspected_items": [],
            },
            rca_cmp_consistency={
                "score": 0.0,
                "assessment": "",
                "inconsistencies": [],
            },
            countermeasure_quality={
                "score": 0.0,
                "assessment": "",
                "weaknesses": [],
            },
            verification_quality={
                "score": 0.0,
                "assessment": "",
                "missing_verifications": [],
            },
            approval_readiness={
                "score": 0.0,
                "assessment": "",
            },
            overall_judgement="needs_review",
            review_reasons=[reason],
            recommended_next_steps=[],
        )


def result_to_dict(result: QualityEvaluationResult | None) -> dict | None:
    return asdict(result) if result else None
