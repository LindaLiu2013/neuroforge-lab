# TUF-E: Self-Learning Agent Framework

**A universal framework for building self-learning agents through pivot emergence, cross-observer consensus, and dissipative knowledge maintenance.**

TUF-E (Thermodynamic Universal Framework for Emergence) is not another reinforcement learning library. It implements a fundamentally different approach to machine learning: instead of optimizing a reward function, the agent learns by maintaining its own viability under thermodynamic constraints. Knowledge emerges as dissipative structures — it exists only while actively maintained, and stops existing when maintenance ceases.

> ## ⚠️ Honest status (read this first)
>
> **This repository is a research framework with an open hypothesis — not a validated learner.**
>
> The theory (pivot three-layer ontology, dissipative knowledge, cross-observer consensus) and the `DomainSpec` interface are the substantive contribution. The **bundled demo domains do NOT yet demonstrate that pivot-based learning beats simple baselines.** An external peer review (Oct 2026) ran the released `TUFEAgent` against trivial controls and found:
>
> - **ArithmeticDomain**: the correct answer is *unobservable by design* (the target is never shown), so viability is capped near 1/3 by chance. A "always-add" control matches the agent.
> - **NKDomain**: the viability band `[0.3, 0.7]` is passed by *random* and *do-nothing* policies (~98%), so the high viability rate proves nothing about learning.
> - **IsingDomain**: the agent's viability is *below* a random-flip baseline — consensus currently hurts in this continuous domain.
> - The released engine's default path is effectively a **state-blind bandit** with a binary viability update; state-conditional pivot machinery exists but is off-by-default or a minority path.
> - "Phase transitions" counted in earlier notes are **Betti-number churn**, not validated topological reorganization.
>
> We are keeping this text public rather than quietly editing it away. See [Known limitations](#known-limitations--what-v11-must-show) and the pre-registered `v1.1` experiment that must make a state-dependent pivot learner beat *all* of: random, no-op, best-fixed-action, and a state-blind bandit — reported together, honestly, even if it loses.

## What makes TUF-E different?

| | Reinforcement Learning | LLMs | TUF-E |
|---|---|---|---|
| **Learning signal** | External reward scalar | Training data distribution | Viability (can I keep going?) |
| **Knowledge** | Policy/value function | Frozen probability distribution | Living dissipative structure |
| **Structure** | Fixed architecture | Fixed at inference | Self-organizing, phase transitions |
| **Forgetting** | Catastrophic or absent | Not applicable | Thermodynamic necessity |
| **Goal** | Maximize reward | Predict next token | Maintain existence |

## Core Concepts

### Pivot (枢机)
The irreducible unit of learning. A pivot is a complete causal transformation: an input pattern, an action, and an output pattern that form a closed loop. Pivots have three inseparable layers:
- **Structure**: the causal transformation itself (端+故)
- **Time**: a complete work cycle (刹那)
- **Function**: maintaining dissipative steady state

### Consensus
Multiple independent observers validate pivots before they become confirmed knowledge. The consensus mechanism auto-adapts to the domain's "consensability" — how well independent observers agree in that domain.

### Dissipation
All knowledge decays. This is not a bug — it's a thermodynamic requirement. Only actively used and reinforced knowledge survives. Dead knowledge is worse than no knowledge because it doesn't adapt.

### Viability
The learning signal is survival, not optimization. "Can I continue to operate effectively?" rather than "Is my score going up?" This prevents reward hacking and drives genuine adaptation.

## Quick Start

```python
from tufe import TUFEAgent
from tufe.examples import ArithmeticDomain

# Pick a built-in domain or create your own
domain = ArithmeticDomain(seed=42)

# Create and run the agent
agent = TUFEAgent(domain, seed=42)
report = agent.run(n_cycles=5000, report_every=500)

print(f"Viability rate: {report['viability_rate']:.3f}")
print(f"Confirmed pivots: {report['knowledge']['n_confirmed']}")
```

## Creating Your Own Domain

Implement the `DomainSpec` interface to apply TUF-E to any problem:

```python
from tufe import DomainSpec, ObserverSpec, ActionResult

class MyDomain(DomainSpec):
    def name(self) -> str:
        return "MyDomain"

    def observers(self) -> list[ObserverSpec]:
        # Return 2+ independent observation channels
        return [MyObserver1(), MyObserver2()]

    def observe(self) -> dict:
        # Return raw observations for each observer
        return {'channel1': raw_data_1, 'channel2': raw_data_2}

    def action_space_size(self) -> int:
        return n_actions

    def act(self, action_id: int) -> ActionResult:
        # Execute action, return result
        return ActionResult(success=ok, post_state={...})

    def assess_viability(self, state: dict) -> bool:
        # Is the agent still functioning effectively?
        return some_condition(state)

    def apply_dissipation(self) -> None:
        # Environmental pressure that erodes agent's work
        pass
```

The six growth conditions (GC-1 through GC-6) from the TUF-E Design Language are encoded as interface requirements. See [the design language document](docs/tufe-design-language-v03.md) for the theoretical foundation.

## Built-in Example Domains

Four reference implementations demonstrate different causal structure types. **Treat their viability rates as unvalidated until read against a baseline** (see [Known limitations](#known-limitations--what-v11-must-show)):

- **ArithmeticDomain**: Discrete exact causality, high consensability. *(Demo caveat: the target is currently unobservable, so viability is chance-capped near 1/3.)*
- **IsingDomain**: Continuous probabilistic causality, low consensability. *(Demo caveat: agent viability currently sits below a random-flip baseline.)*
- **NKDomain**: Combinatorial epistatic causality, medium consensability. *(Demo caveat: the `[0.3, 0.7]` viability band is passed by random/no-op policies.)*
- **RegexDomain** (plus `RegexOpenDomain`, `RegexLoopDomain` GC-1 arms): pattern-construction learning over a 12-action regex builder. This is the domain where the agent is meant to *build* programs; it is also the one that most exposed the open-operation-space question (GC-1).

## Known limitations — what v1.1 must show

The released `v1.0` numbers were reported as viability rates **without baselines**, which made trivially-passable tasks look like learning. The honest comparison the peer review ran (5000 cycles, seed 0):

| Domain | TUF-E | Random | No-op | Best fixed action |
|---|---|---|---|---|
| Arithmetic | 0.305 | 0.136 | 0.020 | 0.339 (always-add) |
| Ising | 0.451 | 0.657 | 0.413 | 0.349 |
| NK | 0.966 | 0.981 | 0.986 | 0.969 |

Read plainly: on Arithmetic the agent is *below* a fixed action; on Ising it is *below* random; on NK everything passes so the number is meaningless. **None of these demonstrate that pivot-based learning works.** That is the gap `v1.1` is pre-registered to close: one domain where a *state-dependent* pivot learner beats random, no-op, best-fixed-action, **and** a state-blind bandit, with all five numbers reported together.

Other engineering corrections folded into this release cycle: the report key is `report['viability_rate']` (a flat float) — not `report['viability']['rate']` as an earlier release note showed; `pyproject.toml` status is now `2 - Pre-Alpha` (it previously and incorrectly claimed `4 - Beta`).

## Installation

```bash
pip install numpy networkx
# Then add this directory to your path or install as package:
pip install -e .
```

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- NetworkX >= 3.0

## Research Background

TUF-E is grounded in:
- **Thermodynamics of computation** (Prigogine dissipative structures)
- **Epistemological constraints** (heterogeneous observation, practice verification)
- **Cross-domain validation** (arithmetic, physics, biology domains)
- **Design Language v0.3** with six growth conditions and four cross-domain laws

See [`docs/tufe-design-language-v03.md`](docs/tufe-design-language-v03.md) for the full theoretical framework.

## License

MIT
