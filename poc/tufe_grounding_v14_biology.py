"""
TUF-E Design Language v0.2 Validation: Biology Domain v14.2
Kauffman NK Boolean Network

Why NK networks:
  - Direct connection to RAF theory (Kauffman/Steel) validated in v1.2
  - Models gene regulatory networks (GRN) — the "metabolism" of early life
  - Discrete state space but CONTINUOUS fitness landscape (bridges arithmetic/Ising gap)
  - Tunable complexity via K parameter (K=2 ordered, K=N chaotic, K~3 critical)
  - Self-organization at criticality is a NATIVE phenomenon here

Domain Mapping:
  GC-1 Open Operation Space -> Toggle any gene (N genes x 2 states = 2N actions)
  GC-2 Heterogeneous Perception -> 3 observers:
           Observer S: State-pattern (which genes are on/off)
           Observer F: Fitness-landscape (current fitness value)
           Observer D: Dynamics (attractor proximity / transient length)
  GC-3 Dissipative Maintenance -> Pivot decay + environmental perturbation
  GC-4 Temporal Structure    -> Observe->Toggle->Update->Observe = work cycle
  GC-5 Feasible Space        -> Maintain viable fitness region (not max fitness!)
  GC-6 Immutable Meta-Layer  -> NK landscape topology + update rules fixed

Key Question: Does pivot+consensus help maintain viability in a biological
fitness landscape where the "right answer" depends on context (epistasis)?

[DRIFT-CHECK] No reward/optimization/maximize. Learning = feasible space maintenance.
"""

import numpy as np
import networkx as nx
import json
import sys
import io
import os
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# NK BOOLEAN NETWORK ENVIRONMENT (GC-6 Immutable Meta-Layer)
# ============================================================
class NKNetwork:
    """
    Kauffman NK model.

    [GC-6] Immutable meta-rules:
      - N genes, each connected to K others (topology fixed at init)
      - Each gene has a random Boolean function of its K inputs
      - Fitness = average of N individual gene contributions
      - Update rule: synchronous Boolean update (all genes update simultaneously)

    [GC-1] Agent can toggle any gene, creating open-ended exploration
    of the 2^N state space.

    The NK landscape has TUNABLE RUGGEDNESS:
      K=1: smooth landscape, single peak (trivial)
      K=N: maximally rugged, random landscape (impossible)
      K=2-3: critical regime, correlated peaks (interesting for learning)
    """

    def __init__(self, N=12, K=3, seed=42):
        self.N = N
        self.K = min(K, N - 1)
        self.rng = np.random.default_rng(seed)

        # Generate random connectivity: each gene connects to K others
        self.connections = []
        for i in range(N):
            possible = [j for j in range(N) if j != i]
            self.rng.shuffle(possible)
            self.connections.append(sorted(possible[:self.K]))

        # Generate random Boolean functions for each gene
        # Each gene has a truth table of size 2^K
        self.truth_tables = []
        for i in range(N):
            table_size = 2 ** self.K
            table = self.rng.integers(0, 2, size=table_size).astype(np.int8)
            self.truth_tables.append(table)

        # Generate fitness contribution tables (continuous values in [0,1])
        self.fitness_tables = []
        for i in range(N):
            table_size = 2 ** self.K
            table = self.rng.random(size=table_size)
            self.fitness_tables.append(table)

        # Initial state: random
        self.state = self.rng.integers(0, 2, size=N).astype(np.int8)
        self.step_count = 0
        self.update_history = []

    def _get_input_index(self, gene_idx):
        """Convert K input bits to truth table index."""
        inputs = self.connections[gene_idx]
        idx = 0
        for bit_pos, inp_gene in enumerate(inputs):
            if self.state[inp_gene]:
                idx |= (1 << bit_pos)
        return idx

    def update(self):
        """Synchronous Boolean update of all genes."""
        new_state = np.zeros(self.N, dtype=np.int8)
        for i in range(self.N):
            input_idx = self._get_input_index(i)
            new_state[i] = self.truth_tables[i][input_idx]
        self.state = new_state
        self.step_count += 1
        return self.state.copy()

    def fitness(self):
        """Compute fitness = mean of individual gene contributions."""
        total = 0.0
        for i in range(self.N):
            input_idx = self._get_input_index(i)
            total += self.fitness_tables[i][input_idx]
        return total / self.N

    def gene_fitness(self, gene_idx):
        """Individual gene's fitness contribution."""
        input_idx = self._get_input_index(gene_idx)
        return float(self.fitness_tables[gene_idx][input_idx])

    def toggle_gene(self, gene_idx):
        """Agent action: flip one gene."""
        self.state[gene_idx] = 1 - self.state[gene_idx]

    def get_state_summary(self):
        f = self.fitness()
        n_active = int(np.sum(self.state))
        return {
            'state': self.state.copy(),
            'n_active': n_active,
            'active_fraction': n_active / self.N,
            'fitness': f,
            'step': self.step_count,
        }

    def attractor_distance_proxy(self, n_steps=10):
        """
        Proxy for distance to nearest attractor.
        Run n_steps of Boolean update and measure state change rate.
        Low change rate = near attractor; High = in transient.
        """
        saved = self.state.copy()
        changes = 0
        for _ in range(n_steps):
            old = self.state.copy()
            self.update()
            changes += np.sum(old != self.state)
        self.state = saved  # restore
        return changes / (n_steps * self.N)

    def apply_perturbation(self, n_genes=1):
        """Environmental perturbation: randomly flip some genes."""
        for _ in range(n_genes):
            g = int(self.rng.integers(0, self.N))
            self.state[g] = 1 - self.state[g]


# ============================================================
# HETEROGENEOUS OBSERVERS (GC-2)
# ============================================================
class ObserverStatePattern:
    """
    Sees WHICH genes are active. Discretizes by activity pattern.
    Blind to fitness values — only sees structure.
    """
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state_summary):
        af = state_summary['active_fraction']
        n_active = state_summary['n_active']
        # Combine activity fraction with parity of active count
        parity_bit = n_active % 2
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]
        level = sum(1 for t in thresholds if af >= t)
        idx = min(level * 2 + parity_bit, 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        # value = active_fraction
        v = max(0.0, min(1.0, value)) if value is not None else 0.5
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]
        level = sum(1 for t in thresholds if v >= t)
        idx = min(level * 2, 15)
        return self.BUCKETS[idx]


class ObserverFitness:
    """
    Sees FITNESS VALUE. Discretizes by fitness level.
    Blind to which specific genes are active — only sees outcome quality.
    """
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state_summary):
        f = state_summary['fitness']
        idx = min(int(f * 16), 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        v = max(0.0, min(1.0, value)) if value is not None else 0.5
        idx = min(int(v * 16), 15)
        return self.BUCKETS[idx]


class ObserverDynamics:
    """
    Sees DYNAMICAL REGIME. Is the system near an attractor or in transient?
    Blind to both state pattern and fitness — only sees stability.
    """
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state_summary, attractor_dist=None):
        if attractor_dist is None:
            attractor_dist = 0.5
        # Low attractor_dist = stable/near attractor
        # High = unstable/in transient
        idx = min(int(attractor_dist * 16), 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        v = max(0.0, min(1.0, value)) if value is not None else 0.5
        idx = min(int(v * 16), 15)
        return self.BUCKETS[idx]


# ============================================================
# CONSENSUS KNOWLEDGE STORE
# ============================================================
class ConsensusKnowledgeStore:
    def __init__(self, min_observers=2, min_strength=0.15):
        self.candidates = {}
        self.confirmed = {}
        self.min_observers = min_observers
        self.min_strength = min_strength

    def propose(self, shared_key, observer_id, strength, action_type):
        if shared_key not in self.candidates:
            self.candidates[shared_key] = {
                'proposals': {}, 'action_type': action_type,
                'total_strength': 0, 'n_proposers': 0
            }
        c = self.candidates[shared_key]
        if observer_id not in c['proposals']:
            c['proposals'][observer_id] = strength
            c['n_proposers'] += 1
            c['total_strength'] += strength
        if c['n_proposers'] >= self.min_observers:
            avg_str = c['total_strength'] / c['n_proposers']
            if avg_str >= self.min_strength:
                self.confirmed[shared_key] = {
                    'strength': avg_str, 'n_confirmers': c['n_proposers'],
                    'action_type': action_type,
                    'confirmers': list(c['proposals'].keys()),
                    'cycle_completions': 0,
                }
                return True
        return False

    def reinforce(self, pivot_key, success):
        if pivot_key in self.confirmed:
            c = self.confirmed[pivot_key]
            if success:
                c['strength'] = min(1.0, c['strength'] + 0.1)
                c['cycle_completions'] = c.get('cycle_completions', 0) + 1
            else:
                c['strength'] = max(0, c['strength'] - 0.03)
                if c['strength'] < self.min_strength:
                    del self.confirmed[pivot_key]

    def decay(self, rate=0.003):
        expired = []
        for k, c in list(self.confirmed.items()):
            c['strength'] -= rate
            if c['strength'] < self.min_strength:
                expired.append(k)
        for k in expired:
            del self.confirmed[k]
        return len(expired)

    def get_confirmed_pivots(self):
        return dict(self.confirmed)


# ============================================================
# FEASIBLE SPACE TRACKER
# ============================================================
class FeasibleSpaceTracker:
    def __init__(self, window_size=20):
        self.volume_history = []
        self.window_size = window_size

    def update(self, step, confirmed_pivots, private_pivots_all):
        active = set()
        for pk in confirmed_pivots:
            if isinstance(pk, tuple) and len(pk) >= 3:
                active.add(str(pk))
        for pp in private_pivots_all:
            for pk, pv in pp.items():
                if pv.get('strength', 0) > 0.3 and pv.get('cycle_completions', 0) >= 1:
                    active.add(str(pk))
        volume = len(active)
        self.volume_history.append({'step': step, 'volume': volume})
        return volume

    def get_report(self):
        volumes = [h['volume'] for h in self.volume_history]
        if not volumes:
            return {'current_volume': 0, 'peak_volume': 0, 'trend': 'no_data'}
        if len(volumes) < self.window_size:
            return {'current_volume': volumes[-1], 'peak_volume': max(volumes), 'trend': 'early'}
        recent = np.mean(volumes[-self.window_size:])
        older_end = len(volumes) - self.window_size
        older_start = max(0, older_end - self.window_size)
        older = np.mean(volumes[older_start:older_end]) if older_end > older_start else np.mean(volumes[:max(older_end, 1)])
        change = (recent - older) / max(older, 1) if older > 0 else (1.0 if recent > 0 else 0.0)
        trend = 'expanding' if change > 0.1 else ('contracting' if change < -0.1 else 'stable')
        return {'current_volume': volumes[-1], 'peak_volume': max(volumes),
                'trend': trend, 'change_rate': round(change, 4)}


# ============================================================
# PHASE TRANSITION DETECTOR
# ============================================================
class PhaseTransitionDetector:
    def __init__(self):
        self.topology_history = []
        self.phase_transitions = []
        self.last_betti = None
        self.last_pivot_keys = set()
        self.criticality_history = []

    def build_knowledge_graph(self, confirmed_pivots):
        G = nx.Graph()
        for pk, pv in confirmed_pivots.items():
            if isinstance(pk, tuple) and len(pk) >= 3:
                in_b, out_b = str(pk[0]), str(pk[2])
                G.add_node(in_b); G.add_node(out_b)
                G.add_edge(in_b, out_b, weight=pv.get('strength', 0.1))
        return G

    def compute_betti(self, G):
        if not G.nodes(): return 0, 0
        b0 = nx.number_connected_components(G)
        b1 = max(0, len(G.edges()) - len(G.nodes()) + b0)
        return b0, b1

    def check(self, step, confirmed, candidates=None):
        G = self.build_knowledge_graph(confirmed)
        b0, b1 = self.compute_betti(G)
        crit = sum(1 for cv in (candidates or {}).values() if cv.get('n_proposers', 0) == 1)
        self.criticality_history.append({'step': step, 'criticality': crit})
        self.topology_history.append({'step': step, 'b0': b0, 'b1': b1,
                                       'nodes': len(G.nodes()), 'edges': len(G.edges())})
        transition = None
        cur_keys = set(confirmed.keys())
        if self.last_betti and (b0 != self.last_betti[0] or b1 != self.last_betti[1]):
            raf = self._raf(confirmed)
            transition = {
                'step': step, 'old_betti': self.last_betti, 'new_betti': (b0, b1),
                'db0': b0 - self.last_betti[0], 'db1': b1 - self.last_betti[1],
                'added': len(cur_keys - self.last_pivot_keys),
                'removed': len(self.last_pivot_keys - cur_keys),
                'criticality': crit, 'raf': raf,
            }
            self.phase_transitions.append(transition)
        self.last_betti = (b0, b1)
        self.last_pivot_keys = cur_keys
        return transition

    def _raf(self, confirmed):
        if not confirmed: return {'is_raf': False, 'coverage': 0.0}
        ins = {str(pk[0]) for pk in confirmed if isinstance(pk, tuple) and len(pk)>=3}
        outs = {str(pk[2]) for pk in confirmed if isinstance(pk, tuple) and len(pk)>=3}
        food = {f'b{i:02d}' for i in range(16)}
        uncov = [i for i in ins if i not in food and i not in outs]
        cov = (len(ins)-len(uncov))/max(len(ins),1)
        return {'is_raf': cov >= 0.75 and len(ins) >= 2, 'coverage': round(cov, 3)}

    def get_report(self):
        crits = [h['criticality'] for h in self.criticality_history]
        return {
            'n_transitions': len(self.phase_transitions),
            'transitions': self.phase_transitions[:20],
            'final_topology': self.topology_history[-1] if self.topology_history else None,
            'crit_mean': round(np.mean(crits), 2) if crits else 0,
        }


# ============================================================
# TUF-E BIOLOGY AGENT
# ============================================================
class TUFEBioAgent:
    """
    Agent learns to maintain viable fitness in NK landscape.

    Actions: Toggle gene i (for i in 0..N-1), plus "do nothing".
    Total actions: N + 1.

    [GC-5] Viability = fitness in viable range [0.3, 0.7].
           NOT maximizing fitness! Maintaining in the "living zone".
           Too low = death; Too high = fragile specialization.
           This mirrors real biology: organisms don't maximize fitness,
           they maintain it above a survival threshold while retaining adaptability.

    [GC-1] Open operation space: toggling different genes creates
    exponentially many possible trajectories through state space.
    """

    def __init__(self, N=12, seed=42):
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.private_pivots = {}
        self.n_actions = N + 1  # toggle each gene + do_nothing
        self.preferences = np.ones(self.n_actions) / self.n_actions
        self.total_actions = 0
        self.viable_count = 0

    def choose_action(self, obs_buckets, confirmed_pivots):
        self.total_actions += 1
        if confirmed_pivots and self.rng.random() < 0.4:
            matching = []
            for pk, pv in confirmed_pivots.items():
                if isinstance(pk, tuple) and len(pk) >= 3:
                    in_pat = str(pk[0])
                    for ob in obs_buckets:
                        if ob == in_pat:
                            matching.append((pk, pv))
                            break
            if matching:
                best = max(matching, key=lambda x: x[1]['strength'])
                aid = best[0][1]
                if isinstance(aid, (int, np.integer)) and 0 <= aid < self.n_actions:
                    return int(aid)
        if self.rng.random() < 0.15:
            return int(self.rng.integers(0, self.n_actions))
        return int(self.rng.choice(self.n_actions, p=self.preferences))

    def execute(self, env, action_id):
        if action_id < self.N:
            env.toggle_gene(action_id)
        # action_id == self.N means "do nothing"
        return action_id

    def assess_viability(self, state_summary):
        """
        [GC-5] Viability = maintaining fitness in survivable range.
        NOT optimization. Survival requires being "good enough", not "best".
        Range [0.3, 0.7] allows adaptability while avoiding extinction.
        """
        f = state_summary['fitness']
        return 0.3 <= f <= 0.7

    def learn(self, pre_state, post_state, action_id, observers, consensus_store,
              pre_attractor_dist, post_attractor_dist):
        viable = self.assess_viability(post_state)
        if viable:
            self.viable_count += 1

        if 0 <= action_id < self.n_actions:
            if viable:
                self.preferences[action_id] *= (1.0 + 0.08)
            else:
                self.preferences[action_id] *= 0.97
            self.preferences /= self.preferences.sum()

        proposed = []
        for obs_name, obs in observers.items():
            if obs_name == 'state_pattern':
                in_val = pre_state['active_fraction']
                out_val = post_state['active_fraction']
            elif obs_name == 'fitness':
                in_val = pre_state['fitness']
                out_val = post_state['fitness']
            else:  # dynamics
                in_val = pre_attractor_dist
                out_val = post_attractor_dist

            in_b = obs.discretize_value(in_val)
            out_b = obs.discretize_value(out_val)
            pivot_key = (in_b, action_id, out_b)

            if viable:
                if pivot_key in self.private_pivots:
                    self.private_pivots[pivot_key]['strength'] = min(1.0,
                        self.private_pivots[pivot_key]['strength'] + 0.15)
                    self.private_pivots[pivot_key]['cycle_completions'] += 1
                else:
                    self.private_pivots[pivot_key] = {
                        'strength': 0.25, 'successes': 1,
                        'uses': 1, 'cycle_completions': 1
                    }
                str_val = self.private_pivots[pivot_key]['strength']
                consensus_store.propose(pivot_key, obs_name, str_val, action_id)
                proposed.append(pivot_key)

        for pk in proposed:
            if pk in consensus_store.confirmed:
                consensus_store.reinforce(pk, viable)
                self.private_pivots.setdefault(pk, {
                    'strength': 0.1, 'successes': 0, 'uses': 0, 'cycle_completions': 0
                })
                self.private_pivots[pk]['strength'] = min(1.0,
                    self.private_pivots[pk]['strength'] + 0.05)

        return viable


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_experiment(N=12, K=3, total_steps=3000, seed=42,
                   use_consensus=True, use_pivots=True):
    env = NKNetwork(N=N, K=K, seed=seed)
    agent = TUFEBioAgent(N=N, seed=seed)
    consensus = ConsensusKnowledgeStore(min_observers=2)
    feasible_tracker = FeasibleSpaceTracker(window_size=20)
    phase_detector = PhaseTransitionDetector()

    observers = {
        'state_pattern': ObserverStatePattern(),
        'fitness': ObserverFitness(),
        'dynamics': ObserverDynamics(),
    }

    history = []
    n_viable = 0
    fitness_history = []

    for step in range(total_steps):
        # Pre-action observation
        pre_state = env.get_state_summary()
        pre_attractor_dist = env.attractor_distance_proxy(n_steps=5)

        # Observers perceive
        obs_buckets = []
        for name, obs in observers.items():
            if name == 'dynamics':
                obs_buckets.append(obs.perceive(pre_state, pre_attractor_dist))
            else:
                obs_buckets.append(obs.perceive(pre_state))

        # Agent acts
        confirmed = consensus.get_confirmed_pivots() if (use_consensus and use_pivots) else {}
        action_id = agent.choose_action(obs_buckets, confirmed)
        agent.execute(env, action_id)

        # Environment updates (Boolean dynamics)
        env.update()

        # Environmental perturbation (DISSIPATION — erodes agent's work)
        # Randomly flip 1 gene every ~10 steps
        if step % 10 == 0:
            env.apply_perturbation(n_genes=1)

        # Post-action observation
        post_state = env.get_state_summary()
        post_attractor_dist = env.attractor_distance_proxy(n_steps=5)
        fitness_history.append(post_state['fitness'])

        # Learn
        if use_pivots:
            store = consensus if use_consensus else ConsensusKnowledgeStore(min_observers=99)
            viable = agent.learn(pre_state, post_state, action_id, observers, store,
                               pre_attractor_dist, post_attractor_dist)
        else:
            viable = agent.assess_viability(post_state)
            agent.total_actions += 1
            if viable:
                agent.viable_count += 1

        if viable:
            n_viable += 1

        if use_consensus and use_pivots:
            consensus.decay(rate=0.003)

        privates = [agent.private_pivots] if use_pivots else [{}]
        conf_now = consensus.get_confirmed_pivots() if (use_consensus and use_pivots) else {}
        vol = feasible_tracker.update(step, conf_now, privates)
        trans = None
        if use_pivots:
            cands = consensus.candidates if use_consensus else None
            trans = phase_detector.check(step, conf_now, cands)

        if step % 100 == 0:
            history.append({
                'step': step,
                'fitness': round(post_state['fitness'], 4),
                'active_frac': round(post_state['active_fraction'], 4),
                'attractor_dist': round(post_attractor_dist, 4),
                'confirmed': len(conf_now), 'volume': vol,
                'phase_transition': trans is not None,
                'viability_rate': round(n_viable / max(step+1, 1), 3),
            })

    stats = {
        'total_steps': total_steps,
        'viability_rate': round(n_viable / max(total_steps, 1), 4),
        'mean_fitness': round(float(np.mean(fitness_history)), 4),
        'fitness_std': round(float(np.std(fitness_history)), 4),
        'total_actions': agent.total_actions,
        'viable_actions': agent.viable_count,
        'n_private_pivots': len(agent.private_pivots),
        'n_confirmed': len(consensus.confirmed) if use_consensus else 0,
        'feasible_space': feasible_tracker.get_report(),
        'phase_transitions': phase_detector.get_report(),
    }

    return {'config': {'N': N, 'K': K, 'steps': total_steps, 'seed': seed,
                       'consensus': use_consensus, 'pivots': use_pivots},
            'history': history, 'stats': stats}


def run_full_validation(n_seeds=5, total_steps=3000):
    print("=" * 70)
    print("TUF-E v0.2 Validation: Biology Domain v14.2 (NK Network)")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Design: 3 conditions x {n_seeds} seeds x {total_steps} steps")
    print(f"NK Network: N=12, K=3 (critical regime)")
    print("=" * 70)

    conditions = {
        'A_pivot_consensus': {'use_consensus': True, 'use_pivots': True},
        'B_pivot_no_consensus': {'use_consensus': False, 'use_pivots': True},
        'C_no_pivot_baseline': {'use_consensus': False, 'use_pivots': False},
    }

    all_results = {}
    for cond_name, params in conditions.items():
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}")
        print(f"{'='*50}")
        seed_results = []
        for s in range(n_seeds):
            sv = 42 + s * 137
            r = run_experiment(N=12, K=3, total_steps=total_steps, seed=sv,
                             use_consensus=params['use_consensus'],
                             use_pivots=params['use_pivots'])
            seed_results.append(r)
            st = r['stats']
            print(f"  Seed {s}: VR={st['viability_rate']:.3f} | "
                  f"Fit={st['mean_fitness']:.3f}+-{st['fitness_std']:.3f} | "
                  f"Pivots={st['n_confirmed']} | "
                  f"PT={st['phase_transitions']['n_transitions']} | "
                  f"Vol={st['feasible_space']['current_volume']}")
        all_results[cond_name] = seed_results

    summary = {}
    for cn, results in all_results.items():
        vrs = [r['stats']['viability_rate'] for r in results]
        fits = [r['stats']['mean_fitness'] for r in results]
        vols = [r['stats']['feasible_space']['current_volume'] for r in results]
        peaks = [r['stats']['feasible_space']['peak_volume'] for r in results]
        npts = [r['stats']['phase_transitions']['n_transitions'] for r in results]
        nconfs = [r['stats']['n_confirmed'] for r in results]
        summary[cn] = {
            'viability_rate': {'mean': round(float(np.mean(vrs)), 4),
                              'std': round(float(np.std(vrs)), 4)},
            'mean_fitness': {'mean': round(float(np.mean(fits)), 4),
                            'std': round(float(np.std(fits)), 4)},
            'feasible_volume': {'mean_current': round(float(np.mean(vols)), 2),
                               'mean_peak': round(float(np.mean(peaks)), 2)},
            'phase_transitions': {'mean_count': round(float(np.mean(npts)), 2),
                                 'total': int(np.sum(npts))},
            'confirmed_pivots': {'mean': round(float(np.mean(nconfs)), 2)},
        }

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    for cn, d in summary.items():
        vr = d['viability_rate']
        fit = d['mean_fitness']
        fv = d['feasible_volume']
        pt = d['phase_transitions']
        print(f"\n  {cn}:")
        print(f"    Viability Rate: {vr['mean']:.3f} +/- {vr['std']:.3f}")
        print(f"    Mean Fitness: {fit['mean']:.3f} +/- {fit['std']:.3f}")
        print(f"    Feasible Volume: {fv['mean_current']:.1f} (peak: {fv['mean_peak']:.1f})")
        print(f"    Phase Transitions: {pt['mean_count']:.1f} avg ({pt['total']} total)")
        print(f"    Confirmed Pivots: {d['confirmed_pivots']['mean']:.1f}")

    print("\n" + "=" * 70)
    print("CROSS-CONDITION ANALYSIS")
    print("=" * 70)
    if 'A_pivot_consensus' in summary and 'C_no_pivot_baseline' in summary:
        a_vr = summary['A_pivot_consensus']['viability_rate']['mean']
        c_vr = summary['C_no_pivot_baseline']['viability_rate']['mean']
        a_fit = summary['A_pivot_consensus']['mean_fitness']['mean']
        c_fit = summary['C_no_pivot_baseline']['mean_fitness']['mean']
        a_vol = summary['A_pivot_consensus']['feasible_volume']['mean_current']
        c_vol = summary['C_no_pivot_baseline']['feasible_volume']['mean_current']
        a_pt = summary['A_pivot_consensus']['phase_transitions']['mean_count']
        print(f"\n  A vs C (pivot+consensus vs baseline):")
        print(f"    VR difference: {a_vr - c_vr:+.3f}")
        print(f"    Fitness difference: {a_fit - c_fit:+.3f}")
        print(f"    Volume difference: {a_vol - c_vol:+.1f}")
        print(f"    Phase transitions: {a_pt:.1f} vs 0")
        if a_vr > c_vr + 0.01:
            print(f"    Verdict: PIVOT+CONSENSUS IMPROVES VIABILITY")
        elif abs(a_vr - c_vr) < 0.01:
            print(f"    Verdict: COMPARABLE")
        else:
            print(f"    Verdict: BASELINE OUTPERFORMS")

    if 'A_pivot_consensus' in summary and 'B_pivot_no_consensus' in summary:
        a_vr = summary['A_pivot_consensus']['viability_rate']['mean']
        b_vr = summary['B_pivot_no_consensus']['viability_rate']['mean']
        a_pt = summary['A_pivot_consensus']['phase_transitions']['mean_count']
        b_pt = summary['B_pivot_no_consensus']['phase_transitions']['mean_count']
        print(f"\n  A vs B (consensus vs no-consensus):")
        print(f"    VR difference: {a_vr - b_vr:+.3f}")
        print(f"    PT difference: {a_pt - b_pt:+.1f}")
        if a_vr > b_vr + 0.01:
            print(f"    Verdict: CONSENSUS ADDS VALUE")
        elif abs(a_vr - b_vr) < 0.01:
            print(f"    Verdict: CONSENSUS NEUTRAL")
        else:
            print(f"    Verdict: NO-CONSENSUS OUTPERFORMS")

    # Cross-domain comparison
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON (Arithmetic v1.2 / Ising v14.1 / Biology v14.2)")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'Arithmetic':>12} {'Ising':>12} {'Biology':>12}")
    print(f"  {'-'*61}")
    a_vr = summary.get('A_pivot_consensus', {}).get('viability_rate', {}).get('mean', 0)
    c_vr = summary.get('C_no_pivot_baseline', {}).get('viability_rate', {}).get('mean', 0)
    print(f"  {'A viability rate':<25} {'0.235':>12} {'0.664':>12} {a_vr:>12.3f}")
    print(f"  {'C viability rate':<25} {'0.228':>12} {'0.687':>12} {c_vr:>12.3f}")
    print(f"  {'A-C difference':<25} {'+0.007':>12} {'-0.023':>12} {a_vr-c_vr:>+12.3f}")
    a_pt = summary.get('A_pivot_consensus', {}).get('phase_transitions', {}).get('mean_count', 0)
    print(f"  {'A phase transitions':<25} {'13.8':>12} {'227.6':>12} {a_pt:>12.1f}")
    a_conf = summary.get('A_pivot_consensus', {}).get('confirmed_pivots', {}).get('mean', 0)
    print(f"  {'A confirmed pivots':<25} {'18':>12} {'57':>12} {a_conf:>12.1f}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, 'v14_biology_validation_results.json')
    output = {
        'experiment': 'TUF-E v0.2 Validation: Biology Domain v14.2 (NK Network)',
        'timestamp': datetime.now().isoformat(),
        'design': 'NK Boolean network N=12 K=3, 3-phase experiment',
        'conditions': summary,
        'gc_mapping': {
            'GC-1': 'Gene toggle actions (N+1 options)',
            'GC-2': '3 observers: state_pattern, fitness, dynamics',
            'GC-3': 'Pivot decay(0.003) + environmental perturbation',
            'GC-4': 'Observe->Toggle->BooleanUpdate->Perturb->Observe = work cycle',
            'GC-5': 'Viability = fitness in [0.3, 0.7] (survival, not optimization)',
            'GC-6': 'NK landscape topology + Boolean update rules immutable',
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

    return summary


if __name__ == '__main__':
    run_full_validation(n_seeds=5, total_steps=3000)
