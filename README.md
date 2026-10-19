# TUF-E: Self-Learning Agent Framework

**A universal framework for building self-learning agents through pivot emergence, cross-observer consensus, and dissipative knowledge maintenance.**

TUF-E (Thermodynamic Universal Framework for Emergence) is not another reinforcement learning library. It implements a fundamentally different approach to machine learning: instead of optimizing a reward function, the agent learns by maintaining its own viability under thermodynamic constraints. Knowledge emerges as dissipative structures — it exists only while actively maintained, and stops existing when maintenance ceases.

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

Three reference implementations demonstrate different causal structure types:

- **ArithmeticDomain**: Discrete exact causality, high consensability
- **IsingDomain**: Continuous probabilistic causality, low consensability
- **NKDomain**: Combinatorial epistatic causality, medium consensability

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
