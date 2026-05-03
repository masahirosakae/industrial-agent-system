# Industrial Agent System

A harness-first multi-agent system for manufacturing, designed to execute industrial workflows with reliability, reproducibility, and evaluation-driven control.

![status](https://img.shields.io/badge/status-PoC-orange)
![stage](https://img.shields.io/badge/stage-Ph.2-lightgrey)
![license](https://img.shields.io/badge/license-TBD-blue)



## Overview

**Industrial Agent System** is a project that decomposes manufacturing workflows into modular “agents” and orchestrates them under a controlled execution framework (Harness).

The primary focus is **not building smarter AI models**, but designing a system that enables:

* reliable execution
* reproducibility
* traceability
* continuous evaluation

The project is currently in a research / proof-of-concept stage.



## Current Status

The following components have been implemented and validated:

* schema v0.2 (AgentInput / AgentOutput / PlanningResult)
* evaluation v0.2 (structure + quality metrics)
* RuleBasedProcessPlanningAgent (deterministic baseline)
* LocalLLMProcessPlanningAgent (Ollama + Qwen2.5 1.5B)
* pytest-based E2E tests (drilling / milling / unknown)
* Parser + Fallback mechanism for LLM output stabilization

Instead of trusting LLM outputs directly, the system ensures correctness via:

```text
LLM → Parser → Fallback → Evaluation
```

This project is currently at **Phase 2 (MVP Agent implementation complete)** and entering **Phase 3 (evaluation system expansion)**.


## Vision

The long-term goal is a **Full Automation Factory** —
a system where manufacturing decisions and operations are autonomously executed by a network of agents.

This project focuses on building the **first reliable step toward that vision**, starting from a practical and verifiable foundation.



## Problem Statement

Manufacturing environments face structural challenges:

* shortage of skilled engineers and operators
* loss of tacit knowledge due to generational shifts
* heavy reliance on human judgment (e.g., drawings, planning, quality)
* limitations of rule-based automation for non-deterministic tasks

These problems require **re-architecting workflows as agent-based systems**, not just adding tools.



## Core Concept

The factory is modeled as a **system of interacting agents**:

| Agent               | Responsibility            |
| ------------------- | ------------------------- |
| Order Agent         | Interpret orders          |
| Planning Agent      | Process planning          |
| Procurement Agent   | Material sourcing         |
| Preparation Agent   | Setup and tooling         |
| Manufacturing Agent | Execution control         |
| Quality Agent       | Quality validation        |
| Shipping Agent      | Shipment decisions        |
| Improvement Agent   | Root cause & optimization |

This decomposition is an initial hypothesis and will evolve.



## Architecture

```text
AgentInput
↓
Agent (Rule-based / LLM)
↓
LLM (optional)
↓
Parser (structure normalization)
↓
Fallback (deterministic correction)
↓
PlanningResult (schema-compliant)
↓
Evaluation Harness
```

This enables:

* model-agnostic design
* evaluation-driven iteration
* reproducible behavior

This architecture separates probabilistic reasoning (LLM) from deterministic control (Harness).


## Harness-First Philosophy

The key design decision:

> **Design the harness before the agent**

LLM-based systems are inherently non-deterministic.
Without constraints, they cannot guarantee reliability.

The Harness provides:

* structured I/O schema
* validation and constraints
* execution control (timeout, retry, fallback)
* evaluation metrics
* logging and traceability
* human-in-the-loop integration

This ensures:

* reproducibility
* traceability
* continuous improvement



## Local LLM Agent (Current Limitation)

The LocalLLMProcessPlanningAgent uses Qwen via Ollama.

However:

* LLM output is not yet reliable
* JSON structure may break
* semantic interpretation is limited

Therefore:

```text
LLM output is NOT treated as ground truth
```

Instead:

1. Parse into schema
2. Apply deterministic fallback
3. Evaluate via harness

This design **absorbs LLM instability at the system level**.

This reflects a design choice to prioritize system-level reliability over model-level performance.


## MVP Scope

* Input: drawing / specification
* Output: structured manufacturing plan
* Target: Planning Agent
* Interface: CLI (planned)
* Evaluation: continuous via harness

Goal:

> Replace part of human decision-making with reproducible assistance



## Testing

Test cases include:

* drilling detection
* milling detection
* unknown input handling

LLM outputs are normalized via parser + fallback before evaluation.



## Why This Matters

Industrial automation requires:

* reproducibility
* traceability
* measurable quality
* controlled failure behavior

Not just "smart AI", but **systems that can be trusted in real industrial environments**.

This project focuses on building **trustworthy systems**, not just intelligent components.



## Roadmap

* improve prompt stability
* strengthen JSON parsing
* introduce fine-tuned models (Unsloth + Qwen)
* expand multi-agent interactions
* extend evaluation metrics toward real-world KPIs



## Positioning

This repository serves as:

* a reference implementation of harness-based AI systems
* a reproducible foundation for industrial AI discussion
* a system design approach for reliable AI agents



## Disclaimer

* This is a PoC project (not production-ready)
* No real industrial data is included
* LLM outputs must always be reviewed by humans
* Use at your own risk



## Final Note

Industrial AI is not about a single powerful model.

It is about:

> structuring workflows, constraining behavior, and continuously evaluating outputs.

Agents will evolve.  
Harnesses will remain.  
And systems that ensure reliability will define real-world AI adoption.

This repository is a commitment to building that foundation.
