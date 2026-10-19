"""
TUF-E: Self-Learning Agent Framework v1.0

A universal framework for building self-learning agents based on the
TUF-E Design Language v0.3. Given any domain specification, the agent
learns through pivot emergence, cross-observer consensus, and dissipative
maintenance — without reward optimization.

Core concepts:
  - Pivot: irreducible causal transformation unit (structure + time + function)
  - Consensus: cross-observer validation of knowledge (adaptive to domain)
  - Dissipation: knowledge has maintenance cost (thermodynamic necessity)
  - Viability: learning signal = "can I keep going?" (not "am I winning?")

Quick start:
    from tufe import TUFEAgent, DomainSpec, ObserverSpec

    class MyDomain(DomainSpec):
        # Implement required methods...
        pass

    agent = TUFEAgent(MyDomain())
    results = agent.run(n_cycles=5000)

Design Language: docs/tufe-design-language-v03.md
Research: https://github.com/[your-org]/tufe
"""

from .domain import DomainSpec, ObserverSpec, ActionResult, DissipationConfig, ConsensusConfig
from .observers import KnowledgeStore, PivotTriplet, estimate_consensability, select_consensus_strategy
from .engine import TUFEAgent, TopologyMonitor, FeasibleSpaceTracker

__version__ = "1.0.0"
__all__ = [
    'TUFEAgent',
    'DomainSpec',
    'ObserverSpec',
    'ActionResult',
    'DissipationConfig',
    'ConsensusConfig',
    'KnowledgeStore',
    'PivotTriplet',
    'TopologyMonitor',
    'FeasibleSpaceTracker',
    'estimate_consensability',
    'select_consensus_strategy',
]
