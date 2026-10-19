"""
TUF-E Prototype 1.0: Domain Specification Interface

This module defines the abstract interface that any domain must implement
to instantiate a TUF-E self-learning agent. It directly encodes GC-1~GC-6
from the TUF-E Design Language v0.3.

A domain is NOT an environment or a benchmark. It is a complete specification
of the constraints under which cognitive structure can self-organize.

Usage:
    class MyDomain(DomainSpec):
        def observe(self) -> dict[str, Any]: ...
        def act(self, action_id: int) -> ActionResult: ...
        def assess_viability(self, state: dict) -> bool: ...
        ...

    agent = TUFEAgent(MyDomain())
    agent.run(n_cycles=10000)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObserverSpec:
    """
    Specification of one heterogeneous perception channel (GC-2).

    Each observer must have a METHODLOGICALLY INDEPENDENT discretization.
    Not just different parameters of the same method — genuinely different
    ways of seeing the world.
    """
    name: str
    description: str
    n_buckets: int = 16

    @abstractmethod
    def discretize(self, raw_observation: Any) -> str:
        """
        Convert raw observation to a discrete bucket label (e.g., 'b07').

        This function is PRIVATE to this observer. No other observer may
        call or inspect it. Independence is enforced by design, not trust.
        """
        ...


@dataclass
class ActionResult:
    """Result of executing one action in the domain."""
    success: bool
    post_state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DissipationConfig:
    """
    Configuration for dissipative maintenance (GC-3).

    Knowledge has a maintenance cost. Without use, it decays.
    The decay rate must be in the critical regime: too low = noise accumulation,
    too high = knowledge network collapse.

    v1.2 parameter sweep found critical range: 0.003~0.01 for arithmetic domain.
    Other domains may differ. CL-1 suggests higher dissipation pressure = more
    valuable pivots.
    """
    pivot_decay_rate: float = 0.003
    min_strength: float = 0.15
    reinforcement_rate: float = 0.1
    penalty_rate: float = 0.03


@dataclass
class ConsensusConfig:
    """
    Configuration for cross-observer consensus (GC-2, CL-2, CL-4).

    In v0.3, consensus strategy adapts to domain consensability.
    High consensability domains: standard threshold-based consensus.
    Low consensability domains: soft consensus or individual aggregation.
    """
    min_observers: int = 2
    min_strength: float = 0.15
    strategy: str = "auto"  # "standard", "soft", "individual", "auto"


class DomainSpec(ABC):
    """
    Abstract domain specification encoding GC-1 through GC-6.

    To create a TUF-E agent for your domain, subclass this and implement
    all abstract methods. The engine handles everything else.

    The six growth conditions map to interface requirements:
      GC-1 (Open Operation Space) → action_space_size() must be > 0,
           and the space should allow unforeseen combinations
      GC-2 (Heterogeneous Perception) → observers() must return ≥ 2
           methodologically independent ObserverSpec instances
      GC-3 (Dissipative Maintenance) → dissipation_config() provides
           decay parameters; apply_dissipation() erodes system state
      GC-4 (Temporal Structure) → work_cycle() defines the complete
           perceive→act→observe loop
      GC-5 (Feasible Space Orientation) → assess_viability() returns
           whether the system can continue operating effectively
      GC-6 (Immutable Meta-Layer) → meta_rules() returns rules that
           the agent cannot modify
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable domain name."""
        ...

    @abstractmethod
    def observers(self) -> list[ObserverSpec]:
        """
        Return ≥ 2 methodologically independent observers (GC-2).

        Each observer sees the world through a different lens.
        Their discretization functions must be genuinely different,
        not just parameter variations of the same approach.
        """
        ...

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """
        Collect raw observations from all channels.

        Returns a dict mapping observer name to raw observation data.
        The engine will pass each entry to the corresponding observer's
        discretize() method.
        """
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        """
        Number of available actions (GC-1).

        Should be large enough to allow unforeseen combinations.
        A closed, fully enumerable space limits learning to search.
        """
        ...

    @abstractmethod
    def act(self, action_id: int) -> ActionResult:
        """
        Execute one action in the domain.

        This is the agent's causal intervention. The action changes
        the world state, and the next observation reflects this change.
        """
        ...

    @abstractmethod
    def assess_viability(self, state: dict[str, Any]) -> bool:
        """
        Is the system still able to operate effectively? (GC-5)

        This is NOT a reward signal. It is a survival check.
        Returns True if the system can continue, False otherwise.

        Critical: this must assess FEASIBILITY (can I keep going?),
        not OPTIMALITY (am I doing as well as possible?).
        """
        ...

    @abstractmethod
    def apply_dissipation(self) -> None:
        """
        Apply domain-specific dissipative forces (GC-3).

        This erodes the system state, creating pressure for the agent
        to maintain viable structure. Without dissipation, pivots have
        no functional value (CL-1).

        Examples: thermal noise, environmental perturbation, resource decay.
        """
        ...

    def dissipation_config(self) -> DissipationConfig:
        """Override to customize dissipation parameters."""
        return DissipationConfig()

    def consensus_config(self) -> ConsensusConfig:
        """Override to customize consensus strategy."""
        return ConsensusConfig()

    def meta_rules(self) -> dict[str, Any]:
        """
        Return immutable meta-layer rules (GC-6).

        These rules cannot be modified by the agent during learning.
        They define what counts as valid knowledge and valid verification.

        Default implementation returns empty dict (no explicit meta-rules
        beyond the engine's built-in constraints).
        """
        return {}

    def initial_perturbation(self) -> None:
        """
        Optional: break initial symmetry to bootstrap learning.

        Called once before the first work cycle. Can randomize state,
        introduce bias, or set up non-trivial initial conditions.

        Default: no perturbation (random initialization is usually sufficient).
        """
        pass
