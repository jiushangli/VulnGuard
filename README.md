# VulnGuard

<div align="center">

**Business Logic Vulnerability Mining & Verification Agent Framework**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A multi-agent framework for white-box code audit, combining decentralized exploration with centralized verification to detect business logic vulnerabilities.*

</div>

---

## Overview

VulnGuard is a specialized agent framework designed for **business logic vulnerability detection and verification** through white-box code analysis. It synthesizes architectural patterns from OpenClaw, Claude Code, and Hermes, while incorporating insights from Cairn, BreachWeave, CodeWiki, and the DELM paper.

### Core Innovation

| Feature | Source | VulnGuard Implementation |
|---------|--------|--------------------------|
| **VulnKB Knowledge Graph** | Cairn + BreachWeave + DELM | Immutable Fact/Intent/Hint DAG with verified admission |
| **Decentralized Mining** | DELM | Task Queue with claim + lease, no central dispatcher |
| **Verified Admission** | DELM paper | Compress → Verify → Admit, prevents error propagation |
| **Failure Boundary** | BreachWeave | Precise "what doesn't work and why" as immutable Facts |
| **API Sequence Analysis** | Original | 4-layer model: endpoint → dependency → sequence constraint → business semantics |
| **Observer Correction** | BreachWeave | Periodic review with cool-down, NO_CHANGE > Adjust > Add |
| **Declarative Tools** | OpenClaw | Risk-level + audit-phase filtering, not hardcoded |
| **Streaming Concurrency** | Claude Code | Safe tools parallel, unsafe tools serial |
| **3-Layer Prompt Cache** | Hermes | Stable/Context/Volatile, byte-stable prefix for cache hit |

## Architecture

```
VulnMiner (Parallel Exploration)          VulnVerifier (Independent Verification)
  ├─ API_SEQUENCE Miner                       ├─ Evidence Chain Validator
  ├─ DATAFLOW_TAINT Miner                     ├─ PoC Constructor
  ├─ BUSINESS_LOGIC Miner                     └─ Boundary Tester
  ├─ ATTACK_SURFACE Miner
  └─ Observer (Strategy Review)           ──────────────────────────────
         │                                       │
         └────────── VulnKB ─────────────────────┘
              Fact (immutable)  Intent (claimed)  Hint (rules)
```

### Audit Pipeline

```
Phase 0: Code Intelligence
  Source → AST Parse → Dependency Graph → Module Cluster → API Sequence → Vuln Hypotheses
  → Initial Facts + Intents → VulnKB

Phase 1: Parallel Vuln Mining
  Miners claim Intents → OODA loops → submit Fact/Intent/FailureBoundary
  Observer reviews every N facts → strategy adjustment (with cool-down)

Phase 2: Independent Verification
  Verifier reads VULN_HYPOTHESIS → 3-phase verify → confirmed / ruled_out / needs_more

Phase 3: Report Generation
  VulnKB → structured audit report (TXT + JSON + SQLite DB + exit code)
```

## Quick Start

```bash
# Full audit
python -m vulnguard audit --target /path/to/repo --language python

# Run specific phase only
python -m vulnguard analyze --target /path/to/repo --phase intelligence

# Generate report from persisted database
python -m vulnguard report --db /path/to/vulnguard.db
```

### Configuration

```yaml
# vulnguard.yaml
target:
  repo_path: /path/to/repo
  language: python
  entry_points: ["app.py", "main.py"]
  exclude_patterns: ["*.test.*", "test_*"]

intelligence:
  max_tokens_per_module: 36000
  max_cluster_depth: 3
  extract_api_routes: true

agents:
  miner_count: 4
  miner_specializations: [api_sequence, dataflow_taint, business_logic, attack_surface]
  verifier_count: 2
  observer_enabled: true

llm:
  providers:
    primary:
      name: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
    verifier:
      name: anthropic
      model: claude-sonnet-4
      api_key: ${ANTHROPIC_API_KEY}
  role_mapping:
    miner: primary
    verifier: verifier    # Independent LLM ensures verification independence
    observer: primary

budget:
  max_ooda_cycles: 20
  heartbeat_interval: 30
```

## Project Structure

```
vulnguard/
├── __init__.py              Package entry
├── __main__.py              CLI: audit / analyze / report
├── agent_base.py            Agent base class + OODA loop (902 lines)
├── orchestrator.py           Top-level orchestrator + report generation (1158 lines)
│
├── vulnkb/
│   └── models.py            VulnKB: Fact/Intent/Hint + verified admission (679 lines)
│
├── tools/
│   ├── descriptor.py         Declarative tool descriptions + availability expressions
│   ├── registry.py           Tool registry + phase-based filtering
│   └── executor.py           Streaming concurrent tool executor
│
├── utils/
│   ├── llm.py                Multi-provider LLM gateway with role routing
│   └── prompt.py             3-layer prompt cache (stable/context/volatile)
│
├── intelligence/
│   ├── parser.py             Multi-language AST parsing (Python/Java/JS/Go)
│   ├── dependency.py         Dependency graph builder
│   ├── module.py             Module clustering (token budget + LLM)
│   ├── api_sequence.py       API sequence graph + vuln hypothesis detection
│   └── engine.py             CodeIntelligenceEngine orchestrator
│
├── miner/
│   └── agent.py              MinerAgent: 6 specialization directions
│
├── observer/
│   └── agent.py              ObserverAgent: strategy review + cool-down
│
├── verifier/
│   └── agent.py              VerifierAgent: 3-phase independent verification
│
└── config/
    └── settings.py           Full configuration system (YAML loading)
```

## What VulnGuard Detects

Through API sequence analysis and business logic reasoning:

| Vulnerability | Detection Method |
|--------------|-----------------|
| **BOLA** (Broken Object-Level Auth) | Resource-dependent API without object-level authorization |
| **BFLA** (Broken Function-Level Auth) | Regular user API reaching admin functionality |
| **State Machine Bypass** | API sequence allows skipping required states |
| **TOCTOU** | State dependency between check and use |
| **Mass Assignment** | Request body without observable whitelist |
| **Injection** | Unsanitized user input reaching interpreters |
| **Auth Bypass** | Missing or weak authentication on protected endpoints |

## Design Philosophy

**"Decentralized exploration + centralized verification"**

- Miners explore independently — no central dispatcher bottleneck
- Verifiers validate independently — different LLM provider ensures no shared hallucination
- VulnKB ensures consistency — verified admission prevents error propagation
- Failure Boundaries are valuable — "what doesn't work" is as important as "what does"

## License

MIT