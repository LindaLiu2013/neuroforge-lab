"""
TUF-E Prototype 1.0: Observer & Consensus Mechanisms

Implements GC-2 (heterogeneous perception independence) and the consensus
mechanism with domain-adaptive strategy selection (CL-2, CL-4).
"""

import numpy as np
from typing import Any


class PivotTriplet:
    """
    A pivot candidate: (input_pattern, action_id, output_pattern).

    This is the structural layer of a pivot — an irreducible causal
    transformation unit. It becomes a confirmed pivot only after
    cross-observer validation (consensus) and temporal verification
    (complete work cycle).
    """
    __slots__ = ('in_bucket', 'action_id', 'out_bucket')

    def __init__(self, in_bucket: str, action_id: int, out_bucket: str):
        self.in_bucket = in_bucket
        self.action_id = action_id
        self.out_bucket = out_bucket

    def key(self) -> tuple:
        return (self.in_bucket, self.action_id, self.out_bucket)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, PivotTriplet):
            return self.key() == other.key()
        return False

    def __repr__(self):
        return f"Pivot({self.in_bucket}→act{self.action_id}→{self.out_bucket})"


class KnowledgeStore:
    """
    Dissipative knowledge store with adaptive consensus (GC-3, CL-2, CL-4).

    Manages both private pivots (individual observer knowledge) and
    confirmed pivots (cross-validated knowledge). Implements decay
    as a thermodynamic necessity, not an engineering feature.
    """

    def __init__(self, min_observers: int = 2, min_strength: float = 0.15,
                 decay_rate: float = 0.003, reinforcement: float = 0.1,
                 penalty: float = 0.03, strategy: str = "standard"):
        self.min_observers = min_observers
        self.min_strength = min_strength
        self.decay_rate = decay_rate
        self.reinforcement = reinforcement
        self.penalty = penalty
        self.strategy = strategy

        # Private knowledge per observer
        self.private: dict[str, dict[tuple, dict]] = {}

        # Consensus candidates (proposed but not yet confirmed)
        self.candidates: dict[tuple, dict] = {}

        # Confirmed pivots (cross-validated)
        self.confirmed: dict[tuple, dict] = {}

    def propose(self, triplet: PivotTriplet, observer_name: str,
                strength: float) -> bool:
        """
        Submit a pivot triplet from one observer for consensus evaluation.

        Returns True if this proposal triggered confirmation.
        """
        key = triplet.key()

        if key not in self.candidates:
            self.candidates[key] = {
                'proposals': {},
                'total_strength': 0.0,
                'n_proposers': 0,
                'action_id': triplet.action_id,
            }

        c = self.candidates[key]
        if observer_name not in c['proposals']:
            c['proposals'][observer_name] = strength
            c['n_proposers'] += 1
            c['total_strength'] += strength

        # Check consensus threshold
        if c['n_proposers'] >= self.min_observers:
            avg_str = c['total_strength'] / c['n_proposers']
            if avg_str >= self.min_strength:
                self.confirmed[key] = {
                    'strength': avg_str,
                    'n_confirmers': c['n_proposers'],
                    'confirmers': list(c['proposals'].keys()),
                    'action_id': triplet.action_id,
                    'cycle_completions': 0,
                }
                # Clean up candidate
                del self.candidates[key]
                return True
        return False

    def reinforce(self, triplet_key: tuple, success: bool) -> None:
        """Reinforce or weaken a confirmed pivot based on outcome."""
        if triplet_key in self.confirmed:
            c = self.confirmed[triplet_key]
            if success:
                c['strength'] = min(1.0, c['strength'] + self.reinforcement)
                c['cycle_completions'] = c.get('cycle_completions', 0) + 1
            else:
                c['strength'] = max(0, c['strength'] - self.penalty)
                if c['strength'] < self.min_strength:
                    del self.confirmed[triplet_key]

    def decay(self) -> int:
        """
        Apply dissipative decay to all confirmed pivots (GC-3).

        Knowledge that isn't maintained dies. This is not forgetting —
        it is thermodynamic screening. Only actively useful knowledge survives.

        Returns number of expired pivots.
        """
        expired = []
        for k, c in list(self.confirmed.items()):
            c['strength'] -= self.decay_rate
            if c['strength'] < self.min_strength:
                expired.append(k)
        for k in expired:
            del self.confirmed[k]
        return len(expired)

    def get_confirmed(self) -> dict[tuple, dict]:
        return dict(self.confirmed)

    def update_private(self, observer_name: str, triplet: PivotTriplet,
                       viable: bool) -> None:
        """Update private pivot store for one observer."""
        if observer_name not in self.private:
            self.private[observer_name] = {}

        key = triplet.key()
        privates = self.private[observer_name]

        if viable:
            if key in privates:
                privates[key]['strength'] = min(1.0, privates[key]['strength'] + 0.15)
                privates[key]['cycle_completions'] += 1
            else:
                privates[key] = {
                    'strength': 0.25,
                    'successes': 1,
                    'uses': 1,
                    'cycle_completions': 1,
                }

    def stats(self) -> dict:
        return {
            'n_confirmed': len(self.confirmed),
            'n_candidates': len(self.candidates),
            'n_private': {name: len(p) for name, p in self.private.items()},
            'mean_strength': round(float(np.mean([c['strength'] for c in self.confirmed.values()])), 3) if self.confirmed else 0.0,
        }


def estimate_consensability(domain, n_samples: int = 200, seed: int = 42) -> float:
    """
    Estimate domain consensability (CL-4).

    Measures how often independent observers agree on the discretization
    of state-action-state transitions. High value = standard consensus works.
    Low value = need soft/individual strategy.

    Returns float in [0, 1].
    """
    rng = np.random.default_rng(seed)
    observers = domain.observers()
    if len(observers) < 2:
        return 1.0

    agreements = 0
    total = 0

    for _ in range(n_samples):
        obs_data = domain.observe()
        buckets = []
        for obs_spec in observers:
            raw = obs_data.get(obs_spec.name, None)
            if raw is not None:
                buckets.append(obs_spec.discretize(raw))

        if len(buckets) >= 2:
            # Check pairwise agreement
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    total += 1
                    if buckets[i] == buckets[j]:
                        agreements += 1

        # Take a random action to change state
        action_id = int(rng.integers(0, domain.action_space_size()))
        domain.act(action_id)

    return agreements / max(total, 1)


def select_consensus_strategy(consensability: float) -> tuple[int, str]:
    """
    Auto-select consensus parameters based on domain consensability (CL-2, CL-4).

    Returns (min_observers, strategy_name).
    """
    if consensability > 0.5:
        return 2, "standard"
    elif consensability > 0.2:
        return 2, "soft"
    else:
        return 1, "individual"
