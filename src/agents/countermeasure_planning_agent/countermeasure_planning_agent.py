import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

# NOTE: This agent depends only on the abstract LLMProvider contract and the
# schema dataclasses exposed by the previous Quality Workflow stages. Concrete
# provider implementations must never be imported here; the provider-agnostic
# test guards this contract.
from src.agents.root_cause_analysis_agent import (
    RootCauseAgentOutput,
    RootCauseAnalysis,
)
from src.llm.base import LLMProvider
from src.llm.factory import create_llm_provider


CountermeasurePlanningStatus = Literal["success", "failure", "needs_review"]


# -------------------------
# Input / Output schema
# -------------------------
@dataclass
class CountermeasurePlanningInput:
    """Input for CountermeasurePlanningAgent.

    Consumes the output of RootCauseAnalysisAgent. Mirrors the
    ``QualityIssueAnalysisAgent -> RootCauseAnalysisAgent`` contract: the
    previous analysis is passed in directly so this agent never re-derives
    facts and only plans on top of validated RCA hypotheses.
    """

    case_id: str
    root_cause_analysis: RootCauseAnalysis
    process_name: str = ""
    product_or_part: str = ""

    @classmethod
    def from_root_cause_output(
        cls,
        root_cause_output: RootCauseAgentOutput,
        process_name: str = "",
        product_or_part: str = "",
    ) -> "CountermeasurePlanningInput":
        if root_cause_output.result is None:
            raise ValueError(
                "RootCauseAgentOutput.result is required to build CountermeasurePlanningInput"
            )

        problems: list[str] = []
        if root_cause_output.status != "success":
            problems.append(f"status={root_cause_output.status!r}")
        if root_cause_output.needs_review:
            problems.append("needs_review=True")
        if not root_cause_output.parse_success:
            problems.append("parse_success=False")
        if not root_cause_output.schema_valid:
            problems.append("schema_valid=False")
        if problems:
            raise ValueError(
                "RootCauseAgentOutput is not a trusted analysis: "
                + ", ".join(problems)
            )

        return cls(
            case_id=root_cause_output.case_id,
            root_cause_analysis=root_cause_output.result,
            process_name=process_name,
            product_or_part=product_or_part,
        )


@dataclass
class CountermeasurePlan:
    containment_actions: list[dict] = field(default_factory=list)
    permanent_actions: list[dict] = field(default_factory=list)
    verification_plan: list[dict] = field(default_factory=list)
    risk_assessment: list[dict] = field(default_factory=list)
    priority_recommendation: list[dict] = field(default_factory=list)
    # ``review_reasons`` is the model-emitted list. The final
    # AgentOutput.review_reasons is derived (model + policy-review merge).
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class CountermeasurePlanningAgentOutput:
    case_id: str
    agent_name: str
    status: CountermeasurePlanningStatus
    result: CountermeasurePlan | None
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
COUNTERMEASURE_PLANNING_OUTPUT_SCHEMA = (
    "containment_actions",
    "permanent_actions",
    "verification_plan",
    "risk_assessment",
    "priority_recommendation",
    "review_reasons",
)

URGENCY_VALUES = {"high", "medium", "low"}
DIFFICULTY_VALUES = {"high", "medium", "low"}
IMPACT_VALUES = {"high", "medium", "low"}
NON_LOW_CONFIDENCES = {"medium", "high"}


class CountermeasurePlanningParseError(ValueError):
    pass


class CountermeasurePlanningSchemaError(ValueError):
    pass


# -------------------------
# Agent
# -------------------------
class CountermeasurePlanningAgent:
    agent_name = "countermeasure_planning_agent"
    system_prompt = (
        "You are an industrial countermeasure planning agent. "
        "Return valid JSON only and never invent root causes that are not "
        "supported by the provided root cause analysis."
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
        self, plan_input: CountermeasurePlanningInput
    ) -> CountermeasurePlanningAgentOutput:
        raw_output = ""
        try:
            prompt = self._build_prompt(plan_input)
            raw_output = self.provider.generate(
                prompt,
                system_prompt=self.system_prompt,
            ).text
        except Exception as e:
            return self._needs_review_output(
                plan_input=plan_input,
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
                plan_input=plan_input,
                error=e,
                error_type="parse_error",
                raw_output=raw_output,
                parse_success=False,
                schema_valid=False,
                note="LLM output could not be parsed; returning conservative review result",
            )

        try:
            self._validate_raw_analysis(data, plan_input.root_cause_analysis)
        except Exception as e:
            return self._needs_review_output(
                plan_input=plan_input,
                error=e,
                error_type="schema_validation_error",
                raw_output=raw_output,
                parse_success=True,
                schema_valid=False,
                note="LLM output failed schema validation; returning conservative review result",
            )

        # Schema is valid. Now derive policy/evidence review reasons and merge
        # them with whatever the model emitted in its own review_reasons.
        result = self._plan_from_valid_data(data)
        model_review_reasons = list(data.get("review_reasons") or [])
        derived_review_reasons = self._policy_review_reasons(
            data, plan_input.root_cause_analysis
        )
        combined_review_reasons = self._merge_review_reasons(
            model_review_reasons, derived_review_reasons
        )

        if combined_review_reasons:
            return CountermeasurePlanningAgentOutput(
                case_id=plan_input.case_id,
                agent_name=self.agent_name,
                status="needs_review",
                result=result,
                confidence=0.5,
                errors=[],
                notes=[
                    "Schema is valid but evidence/policy review is required; "
                    "see review_reasons."
                ],
                needs_review=True,
                review_reasons=combined_review_reasons,
                parse_success=True,
                schema_valid=True,
                error_type="evidence_review_required",
                raw_output=raw_output,
            )

        return CountermeasurePlanningAgentOutput(
            case_id=plan_input.case_id,
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

    @staticmethod
    def _merge_review_reasons(
        model_reasons: list[str], derived_reasons: list[str]
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for reason in list(model_reasons) + list(derived_reasons):
            if not isinstance(reason, str):
                continue
            if reason in seen:
                continue
            merged.append(reason)
            seen.add(reason)
        return merged

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, plan_input: CountermeasurePlanningInput) -> str:
        prior = asdict(plan_input.root_cause_analysis)
        payload = {
            "case_id": plan_input.case_id,
            "process_name": plan_input.process_name,
            "product_or_part": plan_input.product_or_part,
            "root_cause_analysis": prior,
        }

        return f"""
Plan countermeasures on top of the provided root cause analysis and return a structured JSON object only.

# Input
{json.dumps(payload, ensure_ascii=False, indent=2)}

# Reasoning rules
- Separate immediate containment_actions from permanent_actions.
- For hypotheses whose confidence is "low", prefer verification_plan items over permanent_actions.
- If every hypothesis is "low" confidence, permanent_actions MAY be empty as long as verification_plan is non-empty and review_reasons explains why permanent actions are deferred.
- For five_why entries whose evidence_status is "assumed" or "missing", add a verification_plan item that confirms that evidence before relying on it.
- If root_cause_analysis.missing_evidence is non-empty, each evidence item must be reflected in verification_plan[].verification_item or required_data, and the dependent action must be treated as provisional in review_reasons.
- If a hypothesis has its own missing_evidence list, its verification_plan items must also cover those evidences.
- A "low" confidence hypothesis that has a permanent_action MUST also have at least one paired verification_plan item referencing the same hypothesis.
- Make implementation side-effects and operational risks explicit in risk_assessment.
- success_criteria must be measurable / observable so verification is unambiguous.
- Populate review_reasons whenever the plan is provisional (missing evidence, low-confidence hypotheses, contested assumptions); review_reasons MAY be empty only when the plan is fully evidence-backed.

# Output rules
- Return JSON only. No prose, no markdown, no code fences.
- Do not assert causes that are not present in the root cause analysis input.
- urgency, implementation_difficulty, impact must each be one of: high, medium, low.
- permanent_actions[].related_hypothesis and verification_plan[].related_hypothesis must each be one of the hypotheses[].hypothesis strings from the input.
- priority_recommendation[].priority must form an exact 1..N sequence (no duplicates, gaps, reverse order, or 0-start).

# Required JSON schema
{{
  "containment_actions": [
    {{
      "action": "non-empty string",
      "target": "non-empty string",
      "purpose": "non-empty string",
      "urgency": "high|medium|low",
      "owner_candidate": "non-empty string"
    }}
  ],
  "permanent_actions": [
    {{
      "action": "non-empty string",
      "related_hypothesis": "non-empty string matching an input hypothesis",
      "expected_effect": "non-empty string",
      "implementation_difficulty": "high|medium|low"
    }}
  ],
  "verification_plan": [
    {{
      "verification_item": "non-empty string",
      "method": "non-empty string",
      "success_criteria": "non-empty string",
      "required_data": ["non-empty string"],
      "related_hypothesis": "non-empty string matching an input hypothesis"
    }}
  ],
  "risk_assessment": [
    {{
      "risk": "non-empty string",
      "impact": "high|medium|low",
      "mitigation": "non-empty string"
    }}
  ],
  "priority_recommendation": [
    {{"priority": 1, "action": "non-empty string", "reason": "non-empty string"}}
  ],
  "review_reasons": ["string explaining residual uncertainty; [] if none"]
}}
""".strip()

    # -------------------------
    # JSON parsing (strict whole-response only)
    # -------------------------
    def _parse_json_object(self, text: str) -> dict:
        """Parse the entire LLM response as a single JSON object.

        Same policy as :class:`QualityIssueAnalysisAgent` and
        :class:`RootCauseAnalysisAgent`: ``json.loads(text)`` only.
        """
        if not isinstance(text, str):
            raise CountermeasurePlanningParseError("LLM response must be a string")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CountermeasurePlanningParseError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise CountermeasurePlanningParseError(
                "LLM response is not a JSON object"
            )
        return data

    # -------------------------
    # Structural schema validation
    # -------------------------
    def _validate_raw_analysis(
        self, data: dict, rca: RootCauseAnalysis
    ) -> None:
        errors: list[str] = []

        missing_keys = [
            key for key in COUNTERMEASURE_PLANNING_OUTPUT_SCHEMA if key not in data
        ]
        if missing_keys:
            errors.append(f"missing required keys: {missing_keys}")

        valid_hypotheses = {
            h["hypothesis"]
            for h in rca.hypotheses
            if isinstance(h, dict) and isinstance(h.get("hypothesis"), str)
        }

        self._validate_containment_actions(data.get("containment_actions"), errors)
        self._validate_permanent_actions(
            data.get("permanent_actions"), errors, valid_hypotheses
        )
        self._validate_verification_plan(
            data.get("verification_plan"), errors, valid_hypotheses
        )
        self._validate_risk_assessment(data.get("risk_assessment"), errors)
        self._validate_priority_recommendation(
            data.get("priority_recommendation"), errors
        )
        self._validate_review_reasons(data.get("review_reasons"), errors)

        if errors:
            raise CountermeasurePlanningSchemaError("; ".join(errors))

    @staticmethod
    def _validate_object_list_with_enum(
        value,
        list_key: str,
        required_string_fields: tuple[str, ...],
        enum_field: str | None,
        enum_values: set[str] | None,
        errors: list[str],
        require_non_empty: bool = True,
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
            for required_field in required_string_fields:
                item_value = item.get(required_field)
                if not isinstance(item_value, str) or not item_value.strip():
                    errors.append(
                        f"{list_key}[{index}].{required_field} must be a non-empty string"
                    )
            if enum_field is not None and enum_values is not None:
                if item.get(enum_field) not in enum_values:
                    errors.append(
                        f"{list_key}[{index}].{enum_field} must be one of {sorted(enum_values)}"
                    )

    @classmethod
    def _validate_containment_actions(cls, value, errors: list[str]) -> None:
        cls._validate_object_list_with_enum(
            value=value,
            list_key="containment_actions",
            required_string_fields=("action", "target", "purpose", "owner_candidate"),
            enum_field="urgency",
            enum_values=URGENCY_VALUES,
            errors=errors,
        )

    @classmethod
    def _validate_permanent_actions(
        cls, value, errors: list[str], valid_hypotheses: set[str]
    ) -> None:
        # NOTE: empty permanent_actions is structurally allowed; the policy
        # review layer decides whether emptiness is acceptable in context.
        # Specifically, low-confidence-only RCA may justify an empty list,
        # but a non-low-confidence hypothesis demands at least one action.
        cls._validate_object_list_with_enum(
            value=value,
            list_key="permanent_actions",
            required_string_fields=("action", "related_hypothesis", "expected_effect"),
            enum_field="implementation_difficulty",
            enum_values=DIFFICULTY_VALUES,
            errors=errors,
            require_non_empty=False,
        )
        cls._cross_check_related_hypothesis(
            value, "permanent_actions", valid_hypotheses, errors
        )

    @classmethod
    def _validate_verification_plan(
        cls, value, errors: list[str], valid_hypotheses: set[str]
    ) -> None:
        cls._validate_object_list_with_enum(
            value=value,
            list_key="verification_plan",
            required_string_fields=(
                "verification_item",
                "method",
                "success_criteria",
                "related_hypothesis",
            ),
            enum_field=None,
            enum_values=None,
            errors=errors,
        )
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                required_data = item.get("required_data")
                if not isinstance(required_data, list) or not required_data:
                    errors.append(
                        f"verification_plan[{index}].required_data must be a non-empty list"
                    )
                elif any(
                    not isinstance(entry, str) or not entry.strip()
                    for entry in required_data
                ):
                    errors.append(
                        f"verification_plan[{index}].required_data must contain non-empty strings"
                    )
        cls._cross_check_related_hypothesis(
            value, "verification_plan", valid_hypotheses, errors
        )

    @staticmethod
    def _cross_check_related_hypothesis(
        value,
        list_key: str,
        valid_hypotheses: set[str],
        errors: list[str],
    ) -> None:
        """Ensure ``related_hypothesis`` matches one of RCA's hypothesis strings.

        TODO(phase2): introduce a stable ``hypothesis_id`` field on
        RootCauseAnalysis.hypotheses[] so cross-references survive minor
        wording tweaks. Until then we exact-match on the ``hypothesis``
        string, which is brittle to whitespace/punctuation drift.
        """
        if not isinstance(value, list):
            return
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            related = item.get("related_hypothesis")
            if not isinstance(related, str) or not related.strip():
                continue
            if related not in valid_hypotheses:
                errors.append(
                    f"{list_key}[{index}].related_hypothesis must match one of "
                    "the input RCA hypotheses[].hypothesis strings"
                )

    @classmethod
    def _validate_risk_assessment(cls, value, errors: list[str]) -> None:
        cls._validate_object_list_with_enum(
            value=value,
            list_key="risk_assessment",
            required_string_fields=("risk", "mitigation"),
            enum_field="impact",
            enum_values=IMPACT_VALUES,
            errors=errors,
        )

    @staticmethod
    def _validate_priority_recommendation(value, errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("priority_recommendation must be a list")
            return
        if not value:
            errors.append("priority_recommendation must not be empty")
            return
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"priority_recommendation[{index}] must be an object")
                continue
            for key in ("action", "reason"):
                value_str = item.get(key)
                if not isinstance(value_str, str) or not value_str.strip():
                    errors.append(
                        f"priority_recommendation[{index}].{key} must be a non-empty string"
                    )
            priority = item.get("priority")
            if (
                not isinstance(priority, int)
                or isinstance(priority, bool)
                or priority < 1
            ):
                errors.append(
                    f"priority_recommendation[{index}].priority must be a positive integer"
                )

        actual = [
            item.get("priority") if isinstance(item, dict) else None
            for item in value
        ]
        expected = list(range(1, len(value) + 1))
        if actual != expected:
            errors.append(
                "priority_recommendation priorities must be a 1..N sequence; "
                f"got {actual}, expected {expected}"
            )

    @staticmethod
    def _validate_review_reasons(value, errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("review_reasons must be a list")
            return
        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(f"review_reasons[{index}] must be a string")

    # -------------------------
    # Policy / evidence review (runs only after schema is valid)
    # -------------------------
    def _policy_review_reasons(
        self, data: dict, rca: RootCauseAnalysis
    ) -> list[str]:
        reasons: list[str] = []

        confidence_by_hyp = {
            h["hypothesis"]: h.get("confidence")
            for h in rca.hypotheses
            if isinstance(h, dict) and isinstance(h.get("hypothesis"), str)
        }
        non_low_hypotheses = {
            h for h, c in confidence_by_hyp.items() if c in NON_LOW_CONFIDENCES
        }
        all_hypotheses_low = (
            bool(confidence_by_hyp) and not non_low_hypotheses
        )

        permanent_actions = data.get("permanent_actions") or []
        verification_plan = data.get("verification_plan") or []
        model_review_reasons = data.get("review_reasons") or []

        verification_by_hyp: dict[str, list[dict]] = {}
        for item in verification_plan:
            if not isinstance(item, dict):
                continue
            related = item.get("related_hypothesis")
            if isinstance(related, str):
                verification_by_hyp.setdefault(related, []).append(item)

        all_verification_text = self._verification_text(verification_plan)

        # Rule: empty permanent_actions handling.
        if not permanent_actions:
            if non_low_hypotheses:
                reasons.append(
                    "permanent_actions is empty but non-low-confidence "
                    f"hypotheses exist: {sorted(non_low_hypotheses)}"
                )
            elif all_hypotheses_low:
                model_text = " ".join(
                    r for r in model_review_reasons if isinstance(r, str)
                ).lower()
                if "low" not in model_text:
                    reasons.append(
                        "permanent_actions is empty under low-confidence-only "
                        "hypotheses; review_reasons must explain why permanent "
                        "actions are deferred"
                    )

        # Rule: low-confidence hypothesis with a permanent_action requires a
        # paired verification_plan item for the same hypothesis.
        for action in permanent_actions:
            if not isinstance(action, dict):
                continue
            related = action.get("related_hypothesis")
            if (
                isinstance(related, str)
                and confidence_by_hyp.get(related) == "low"
                and related not in verification_by_hyp
            ):
                reasons.append(
                    f"permanent_action for low-confidence hypothesis "
                    f"'{related}' lacks a paired verification_plan item"
                )

        # Rule: RCA top-level missing_evidence must appear somewhere in the
        # verification_plan (verification_item / required_data substring).
        for me_item in rca.missing_evidence:
            if not isinstance(me_item, dict):
                continue
            evidence = me_item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                continue
            if evidence.lower() not in all_verification_text:
                reasons.append(
                    f"RCA missing_evidence '{evidence}' is not reflected in "
                    "verification_plan"
                )

        # Rule: hypothesis-level missing_evidence must appear in that
        # hypothesis's own verification_plan items.
        for hyp in rca.hypotheses:
            if not isinstance(hyp, dict):
                continue
            hyp_text = hyp.get("hypothesis")
            if not isinstance(hyp_text, str):
                continue
            hyp_evidence = hyp.get("missing_evidence") or []
            hyp_verifications = verification_by_hyp.get(hyp_text, [])
            hyp_text_blob = self._verification_text(hyp_verifications)
            for evidence in hyp_evidence:
                if not isinstance(evidence, str) or not evidence.strip():
                    continue
                if evidence.lower() not in hyp_text_blob:
                    reasons.append(
                        f"hypothesis '{hyp_text}' missing_evidence "
                        f"'{evidence}' is not reflected in its verification_plan"
                    )

        # Rule: five_why entries with evidence_status in {assumed, missing}
        # must be reflected in at least one verification_plan item. Coverage
        # is approximated by keyword overlap (length->=5 alphabetic tokens)
        # between the entry's answer/why text and the verification text blob.
        for entry in rca.five_why:
            if not isinstance(entry, dict):
                continue
            status = entry.get("evidence_status")
            if status not in ("assumed", "missing"):
                continue
            # Use only the ``answer`` text for coverage matching. The ``why``
            # question is the prompt being asked and often contains generic
            # process terms that already appear in unrelated verification
            # items; matching on it would silently approve plans that do not
            # actually verify the unconfirmed answer.
            answer = entry.get("answer")
            keywords: set[str] = set()
            if isinstance(answer, str):
                keywords.update(
                    token.lower()
                    for token in re.findall(r"[A-Za-z]{5,}", answer)
                )
            if not keywords:
                continue
            if not any(kw in all_verification_text for kw in keywords):
                reasons.append(
                    f"five_why level={entry.get('level')} evidence_status="
                    f"{status} is not reflected in verification_plan"
                )

        return reasons

    @staticmethod
    def _verification_text(items) -> str:
        parts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("verification_item", "method", "success_criteria"):
                value = item.get(key)
                if isinstance(value, str):
                    parts.append(value)
            required_data = item.get("required_data") or []
            for entry in required_data:
                if isinstance(entry, str):
                    parts.append(entry)
        return " ".join(parts).lower()

    # -------------------------
    # Build dataclass / fallback
    # -------------------------
    @staticmethod
    def _plan_from_valid_data(data: dict) -> CountermeasurePlan:
        return CountermeasurePlan(
            containment_actions=data["containment_actions"],
            permanent_actions=data["permanent_actions"],
            verification_plan=data["verification_plan"],
            risk_assessment=data["risk_assessment"],
            priority_recommendation=data["priority_recommendation"],
            review_reasons=list(data.get("review_reasons") or []),
        )

    def _needs_review_output(
        self,
        plan_input: CountermeasurePlanningInput,
        error: Exception,
        error_type: str,
        raw_output: str,
        parse_success: bool,
        schema_valid: bool,
        note: str,
    ) -> CountermeasurePlanningAgentOutput:
        reason = f"{error_type}: {error}"
        return CountermeasurePlanningAgentOutput(
            case_id=plan_input.case_id,
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
    def _fallback_result() -> CountermeasurePlan:
        # Conservative empty plan envelope. Human review is the only safe
        # next action: we never inject containment/permanent/verification
        # items here, because the model output was unavailable, malformed,
        # or did not match the planning contract.
        return CountermeasurePlan(
            containment_actions=[],
            permanent_actions=[],
            verification_plan=[],
            risk_assessment=[],
            priority_recommendation=[],
            review_reasons=[],
        )


def plan_to_dict(plan: CountermeasurePlan | None) -> dict | None:
    return asdict(plan) if plan else None
