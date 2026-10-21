"""
TUF-E Prototype 1.0: Universal Pivot Engine

The core learning loop that works across ANY domain implementing DomainSpec.
This engine does not know about arithmetic, physics, or biology — it only
knows about pivots, consensus, dissipation, and viability.

The work cycle (GC-4):
    1. Perceive (all observers independently discretize)
    2. Choose action (pivot-guided or exploratory)
    3. Execute action (domain-specific causal intervention)
    4. Apply dissipation (environmental erosion)
    5. Re-perceive (observe post-action state)
    6. Assess viability (GC-5: can the system continue?)
    7. Update knowledge (propose pivots, reinforce, decay)
    8. Monitor topology (detect phase transitions)
"""

import numpy as np
import networkx as nx
from typing import Any, Optional

from .domain import DomainSpec, ActionResult
from .observers import KnowledgeStore, PivotTriplet, estimate_consensability, select_consensus_strategy


class TopologyMonitor:
    """
    Monitors knowledge graph topology for phase transitions (OS-1).

    Uses Betti numbers as topological invariants. Changes in Betti numbers
    indicate structural reorganization of the knowledge network — potential
    "insight" events.
    """

    def __init__(self):
        self.history: list[dict] = []
        self.transitions: list[dict] = []
        self.last_betti: Optional[tuple] = None
        self.last_keys: set = set()

    def check(self, cycle: int, confirmed: dict) -> Optional[dict]:
        G = nx.Graph()
        for pk, pv in confirmed.items():
            if isinstance(pk, tuple) and len(pk) >= 3:
                in_b, out_b = str(pk[0]), str(pk[2])
                G.add_node(in_b)
                G.add_node(out_b)
                G.add_edge(in_b, out_b, weight=pv.get('strength', 0.1))

        if not G.nodes():
            b0, b1 = 0, 0
        else:
            b0 = nx.number_connected_components(G)
            b1 = max(0, len(G.edges()) - len(G.nodes()) + b0)

        record = {
            'cycle': cycle, 'b0': b0, 'b1': b1,
            'nodes': len(G.nodes()), 'edges': len(G.edges()),
        }
        self.history.append(record)

        transition = None
        cur_keys = set(confirmed.keys())
        if self.last_betti and (b0 != self.last_betti[0] or b1 != self.last_betti[1]):
            raf = self._check_raf(confirmed)
            transition = {
                'cycle': cycle,
                'old_betti': self.last_betti,
                'new_betti': (b0, b1),
                'pivots_added': len(cur_keys - self.last_keys),
                'pivots_removed': len(self.last_keys - cur_keys),
                'raf_closure': raf,
            }
            self.transitions.append(transition)

        self.last_betti = (b0, b1)
        self.last_keys = cur_keys
        return transition

    def _check_raf(self, confirmed: dict) -> dict:
        if not confirmed:
            return {'is_raf': False, 'coverage': 0.0}
        ins = {str(pk[0]) for pk in confirmed if isinstance(pk, tuple) and len(pk) >= 3}
        outs = {str(pk[2]) for pk in confirmed if isinstance(pk, tuple) and len(pk) >= 3}
        # Food set = all possible bucket labels (environment provides any input)
        food = set()
        for i in range(16):
            food.add(f'b{i:02d}')
        uncov = [i for i in ins if i not in food and i not in outs]
        cov = (len(ins) - len(uncov)) / max(len(ins), 1)
        return {'is_raf': cov >= 0.75 and len(ins) >= 2, 'coverage': round(cov, 3)}

    def stats(self) -> dict:
        return {
            'n_transitions': len(self.transitions),
            'transitions': self.transitions[-10:],
            'final_topology': self.history[-1] if self.history else None,
        }


class FeasibleSpaceTracker:
    """
    Tracks feasible space volume over time (GC-5, OS-2).

    Feasible space = set of known viable state-action-state transitions.
    Volume changes indicate learning (expansion) or degradation (contraction).
    """

    def __init__(self, window_size: int = 20):
        self.history: list[dict] = []
        self.window_size = window_size

    def update(self, cycle: int, confirmed: dict, private_all: list[dict]) -> int:
        active = set()
        for pk in confirmed:
            if isinstance(pk, tuple) and len(pk) >= 3:
                active.add(str(pk))
        for pp in private_all:
            for pk, pv in pp.items():
                if pv.get('strength', 0) > 0.3 and pv.get('cycle_completions', 0) >= 1:
                    active.add(str(pk))

        volume = len(active)
        self.history.append({'cycle': cycle, 'volume': volume})
        return volume

    def trend(self) -> tuple[str, float]:
        volumes = [h['volume'] for h in self.history]
        if len(volumes) < self.window_size * 2:
            return 'early', 0.0
        recent = np.mean(volumes[-self.window_size:])
        older = np.mean(volumes[-2*self.window_size:-self.window_size])
        change = (recent - older) / max(older, 1) if older > 0 else (1.0 if recent > 0 else 0.0)
        label = 'expanding' if change > 0.1 else ('contracting' if change < -0.1 else 'stable')
        return label, round(change, 4)

    def stats(self) -> dict:
        volumes = [h['volume'] for h in self.history]
        t, r = self.trend()
        return {
            'current_volume': volumes[-1] if volumes else 0,
            'peak_volume': max(volumes) if volumes else 0,
            'trend': t,
            'change_rate': r,
        }


class TUFEAgent:
    """
    Universal TUF-E Self-Learning Agent.

    Given any DomainSpec implementation, this agent will:
      - Learn domain-specific pivots through cross-observer consensus
      - Maintain knowledge against dissipative forces
      - Adapt its consensus strategy to domain consensability
      - Report observable signs of cognitive growth (OS-1~OS-4)

    The agent does NOT optimize a reward function. It maintains viability.
    """

    def __init__(self, domain: DomainSpec, seed: int = 42, exact_addressing: bool = False):
        self.domain = domain
        self.rng = np.random.default_rng(seed)

        # SC-1 state-sufficient causal memory (v1.2 experiment channel)
        self.exact_addressing = exact_addressing
        self.exact_mem: dict[str, dict[int, tuple[int, int]]] = {}
        self.exact_fires = 0
        self.exact_fire_viable = 0
        self._exact_fire_pending = False

        # Auto-detect consensability and configure consensus (CL-2, CL-4)
        cons_config = domain.consensus_config()
        if cons_config.strategy == "auto":
            try:
                consensability = estimate_consensability(domain, n_samples=100, seed=seed)
            except Exception:
                consensability = 0.5
            min_obs, strategy = select_consensus_strategy(consensability)
            self.consensability = consensability
        else:
            min_obs = cons_config.min_observers
            strategy = cons_config.strategy
            self.consensability = None

        diss_config = domain.dissipation_config()

        self.knowledge = KnowledgeStore(
            min_observers=min_obs,
            min_strength=cons_config.min_strength,
            decay_rate=diss_config.pivot_decay_rate,
            reinforcement=diss_config.reinforcement_rate,
            penalty=diss_config.penalty_rate,
            strategy=strategy,
        )

        self.topology = TopologyMonitor()
        self.feasible_space = FeasibleSpaceTracker()

        # Action preferences (initialized uniform)
        n_actions = domain.action_space_size()
        self.preferences = np.ones(n_actions) / n_actions

        # Statistics
        self.total_cycles = 0
        self.viable_cycles = 0
        self.cycle_log: list[dict] = []

    def choose_action(self, obs_buckets: list[str], state_key: str | None = None) -> int:
        """
        Select action using pivot guidance or exploration.

        If confirmed pivots match current observation, use them (40% prob).
        Otherwise, sample from learned preferences with exploration noise.
        With exact_addressing, a state-sufficient causal memory replays
        first (SC-1): the full state key addresses (state -> action) records
        whose viability rate is already established.
        """
        n_actions = self.domain.action_space_size()

        if self.exact_addressing and state_key is not None:
            rec = self.exact_mem.get(state_key)
            if rec:
                candidates = [(v / n, aid) for aid, (n, v) in rec.items()
                              if n >= 3 and v >= 0.5 * n]
                if candidates and self.rng.random() < 0.8:
                    self.exact_fires += 1
                    self._exact_fire_pending = True
                    return int(max(candidates)[1])

        confirmed = self.knowledge.get_confirmed()

        # Pivot-guided selection
        if confirmed and self.rng.random() < 0.4:
            matching = []
            for pk, pv in confirmed.items():
                if isinstance(pk, tuple) and len(pk) >= 3:
                    in_pat = str(pk[0])
                    for ob in obs_buckets:
                        if ob == in_pat:
                            matching.append((pk, pv))
                            break
            if matching:
                best = max(matching, key=lambda x: x[1]['strength'])
                aid = best[0][1]
                if isinstance(aid, (int, np.integer)) and 0 <= aid < n_actions:
                    return int(aid)

        # Exploration vs exploitation
        if self.rng.random() < 0.15:
            return int(self.rng.integers(0, n_actions))
        return int(self.rng.choice(n_actions, p=self.preferences))

    def run_cycle(self) -> dict:
        """
        Execute one complete work cycle (GC-4).

        Returns cycle summary including viability, pivot updates, and
        any detected phase transitions.
        """
        cycle = self.total_cycles
        self.total_cycles += 1

        # Step 1: Perceive
        raw_obs = self.domain.observe()
        observer_specs = self.domain.observers()
        obs_buckets = []
        for obs_spec in observer_specs:
            raw = raw_obs.get(obs_spec.name, None)
            if raw is not None:
                obs_buckets.append(obs_spec.discretize(raw))
            else:
                obs_buckets.append('b00')
        state_key = repr(sorted((k, repr(v)) for k, v in raw_obs.items()))

        # Step 2: Choose action
        action_id = self.choose_action(obs_buckets, state_key)

        # Step 3: Execute
        result = self.domain.act(action_id)

        # Step 4: Dissipation
        self.domain.apply_dissipation()

        # Step 5: Re-perceive
        post_raw = self.domain.observe()
        post_buckets = []
        for obs_spec in observer_specs:
            raw = post_raw.get(obs_spec.name, None)
            if raw is not None:
                post_buckets.append(obs_spec.discretize(raw))
            else:
                post_buckets.append('b00')

        # Step 6: Assess viability (GC-5)
        viable = self.domain.assess_viability(result.post_state)
        if viable:
            self.viable_cycles += 1
        if self._exact_fire_pending:
            self.exact_fire_viable += 1 if viable else 0
            self._exact_fire_pending = False
        if self.exact_addressing:
            rec = self.exact_mem.setdefault(state_key, {})
            n, v = rec.get(action_id, (0, 0))
            rec[action_id] = (n + 1, v + (1 if viable else 0))

        # Step 7: Update knowledge
        # Update preferences based on viability (NOT reward)
        n_actions = self.domain.action_space_size()
        if 0 <= action_id < n_actions:
            if viable:
                self.preferences[action_id] *= (1.0 + 0.08)
            else:
                self.preferences[action_id] *= 0.97
            self.preferences /= self.preferences.sum()

        # Build pivot triplets from each observer
        for i, obs_spec in enumerate(observer_specs):
            in_b = obs_buckets[i] if i < len(obs_buckets) else 'b00'
            out_b = post_buckets[i] if i < len(post_buckets) else 'b00'
            triplet = PivotTriplet(in_b, action_id, out_b)

            # Update private store
            self.knowledge.update_private(obs_spec.name, triplet, viable)

            # Propose for consensus (only if viable)
            if viable:
                priv = self.knowledge.private.get(obs_spec.name, {})
                strength = priv.get(triplet.key(), {}).get('strength', 0.25)
                self.knowledge.propose(triplet, obs_spec.name, strength)

        # Reinforce confirmed pivots used this cycle
        for pk in list(self.knowledge.confirmed.keys()):
            if isinstance(pk, tuple) and len(pk) >= 3 and pk[1] == action_id:
                self.knowledge.reinforce(pk, viable)

        # Decay
        n_expired = self.knowledge.decay()

        # Step 8: Monitor topology
        confirmed_now = self.knowledge.get_confirmed()
        transition = self.topology.check(cycle, confirmed_now)

        # Track feasible space
        private_all = list(self.knowledge.private.values())
        volume = self.feasible_space.update(cycle, confirmed_now, private_all)

        cycle_summary = {
            'cycle': cycle,
            'viable': viable,
            'action_id': action_id,
            'confirmed_pivots': len(confirmed_now),
            'feasible_volume': volume,
            'phase_transition': transition is not None,
            'expired_pivots': n_expired,
        }

        self.cycle_log.append(cycle_summary)
        return cycle_summary

    def run(self, n_cycles: int = 1000, report_every: int = 100) -> dict:
        """
        Run the agent for n_cycles work cycles.

        Returns comprehensive results including statistics, phase transitions,
        and feasible space evolution.
        """
        print(f"TUF-E Agent starting on domain: {self.domain.name()}")
        print(f"  Observers: {[o.name for o in self.domain.observers()]}")
        print(f"  Action space: {self.domain.action_space_size()}")
        print(f"  Consensus strategy: {self.knowledge.strategy}")
        if self.consensability is not None:
            print(f"  Estimated consensability: {self.consensability:.3f}")
        print(f"  Running {n_cycles} cycles...")
        print()

        self.domain.initial_perturbation()

        for c in range(n_cycles):
            summary = self.run_cycle()

            if (c + 1) % report_every == 0:
                vr = self.viable_cycles / max(c + 1, 1)
                ks = self.knowledge.stats()
                fs = self.feasible_space.stats()
                ts = self.topology.stats()
                print(f"  Cycle {c+1:>5}/{n_cycles}: "
                      f"VR={vr:.3f} | "
                      f"Pivots={ks['n_confirmed']} | "
                      f"Vol={fs['current_volume']} | "
                      f"PT={ts['n_transitions']} | "
                      f"Trend={fs['trend']}")

        # Final report
        final = self.report()
        print()
        print("=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(f"  Domain: {self.domain.name()}")
        print(f"  Total cycles: {self.total_cycles}")
        print(f"  Viability rate: {final['viability_rate']:.3f}")
        print(f"  Confirmed pivots: {final['knowledge']['n_confirmed']}")
        print(f"  Phase transitions: {final['topology']['n_transitions']}")
        print(f"  Feasible space: {final['feasible_space']['current_volume']} "
              f"({final['feasible_space']['trend']})")
        print(f"  Consensus strategy: {self.knowledge.strategy}")
        if self.consensability is not None:
            print(f"  Consensability: {self.consensability:.3f}")

        return final

    def report(self) -> dict:
        """Generate comprehensive experiment report."""
        return {
            'domain': self.domain.name(),
            'total_cycles': self.total_cycles,
            'viable_cycles': self.viable_cycles,
            'viability_rate': round(self.viable_cycles / max(self.total_cycles, 1), 4),
            'knowledge': self.knowledge.stats(),
            'feasible_space': self.feasible_space.stats(),
            'topology': self.topology.stats(),
            'consensus': {
                'strategy': self.knowledge.strategy,
                'min_observers': self.knowledge.min_observers,
                'consensability': self.consensability,
            },
            'meta_rules': self.domain.meta_rules(),
        }
