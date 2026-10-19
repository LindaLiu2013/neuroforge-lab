"""
TUF-E Design Language v0.2 Validation: Ising Domain v14.1

REVISED: Agent is PRIMARY DRIVER of system dynamics.
Environment provides thermal noise AFTER agent action (dissipative erosion).
No automatic Metropolis sweeps — agent IS the dynamics.

Domain Mapping:
  GC-1 Open Operation Space -> Flip any site (64 sites x 2 dirs + do_nothing = 129)
  GC-2 Heterogeneous Perception -> 3 observers (magnetization/correlation/energy)
  GC-3 Dissipative Maintenance -> Pivot decay + thermal perturbation erodes order
  GC-4 Temporal Structure    -> Observe->Act->Noise->Observe = work cycle
  GC-5 Feasible Space        -> Maintain non-trivial structure against thermal erosion
  GC-6 Immutable Meta-Layer  -> Ising Hamiltonian fixed, Boltzmann noise fixed

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


class IsingEnvironment:
    def __init__(self, L=8, J=1.0, seed=42):
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(L, L)).astype(np.int8)
        self.step_count = 0

    @property
    def N(self):
        return self.L * self.L

    def magnetization(self):
        return float(np.sum(self.spins)) / self.N

    def energy(self):
        E = 0.0
        for i in range(self.L):
            for j in range(self.L):
                s = self.spins[i, j]
                right = self.spins[i, (j + 1) % self.L]
                down = self.spins[(i + 1) % self.L, j]
                E -= self.J * s * (right + down)
        return E

    def correlation_proxy(self):
        agreements = 0
        total = 0
        for i in range(self.L):
            for j in range(self.L):
                s = self.spins[i, j]
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = (i + di) % self.L, (j + dj) % self.L
                    if s == self.spins[ni, nj]:
                        agreements += 1
                    total += 1
        return agreements / max(total, 1)

    def local_field(self, i, j):
        return (self.spins[i, (j+1)%self.L] + self.spins[i, (j-1)%self.L] +
                self.spins[(i+1)%self.L, j] + self.spins[(i-1)%self.L, j])

    def apply_thermal_noise(self, temperature):
        if temperature <= 0:
            return 0
        n_flipped = 0
        for _ in range(self.N // 2):
            i = int(self.rng.integers(0, self.L))
            j = int(self.rng.integers(0, self.L))
            delta_e = 2.0 * self.J * self.spins[i, j] * self.local_field(i, j)
            if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / temperature):
                self.spins[i, j] *= -1
                n_flipped += 1
        return n_flipped

    def get_state(self):
        mag = self.magnetization()
        return {
            'magnetization': mag,
            'abs_magnetization': abs(mag),
            'energy': self.energy(),
            'energy_per_site': self.energy() / self.N,
            'correlation': self.correlation_proxy(),
            'step': self.step_count,
        }


class ObserverMagnetization:
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state):
        mag = state['magnetization']
        av = abs(mag)
        sign_bit = 0 if mag >= 0 else 1
        thresholds = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9]
        level = sum(1 for t in thresholds if av >= t)
        idx = min(level + sign_bit * 8, 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        av = abs(value) if value is not None else 0
        sign_bit = 0 if (value or 0) >= 0 else 1
        thresholds = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9]
        level = sum(1 for t in thresholds if av >= t)
        idx = min(level + sign_bit * 8, 15)
        return self.BUCKETS[idx]


class ObserverCorrelation:
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state):
        corr = state.get('correlation', 0.5)
        norm = max(0.0, min(1.0, (corr - 0.3) / 0.7))
        idx = min(int(norm * 16), 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        norm = max(0.0, min(1.0, (value - 0.3) / 0.7)) if value is not None else 0.5
        idx = min(int(norm * 16), 15)
        return self.BUCKETS[idx]


class ObserverEnergy:
    BUCKETS = [f'b{i:02d}' for i in range(16)]

    def perceive(self, state):
        eps = state.get('energy_per_site', -1.0)
        norm = max(0.0, min(1.0, (eps + 2.0) / 2.0))
        idx = min(int(norm * 16), 15)
        return self.BUCKETS[idx]

    def discretize_value(self, value):
        norm = max(0.0, min(1.0, (value + 2.0) / 2.0)) if value is not None else 0.5
        idx = min(int(norm * 16), 15)
        return self.BUCKETS[idx]


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


class TUFEIsingAgent:
    def __init__(self, L=8, seed=42):
        self.L = L
        self.N = L * L
        self.rng = np.random.default_rng(seed)
        self.private_pivots = {}
        self.n_actions = self.N * 2 + 1
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
        if action_id >= self.N * 2:
            return action_id
        site = action_id // 2
        direction = action_id % 2
        i, j = site // self.L, site % self.L
        target = 1 if direction == 0 else -1
        env.spins[i, j] = target
        return action_id

    def assess_viability(self, state):
        am = state['abs_magnetization']
        corr = state.get('correlation', 0.5)
        eps = state.get('energy_per_site', -1.0)
        return (0.1 <= am <= 0.8) and (corr >= 0.4) and (eps > -1.8)

    def learn(self, pre_state, post_state, action_id, observers, consensus_store):
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
            if obs_name == 'magnitude':
                in_val, out_val = pre_state['magnetization'], post_state['magnetization']
            elif obs_name == 'correlation':
                in_val = pre_state.get('correlation', 0.5)
                out_val = post_state.get('correlation', 0.5)
            else:
                in_val = pre_state.get('energy_per_site', -1.0)
                out_val = post_state.get('energy_per_site', -1.0)
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


def temperature_schedule(step, total_steps, T_critical=2.269):
    p = step / total_steps
    if p < 0.3:
        return 4.0 - (p / 0.3) * (4.0 - T_critical)
    elif p < 0.6:
        t_local = (p - 0.3) / 0.3
        return T_critical + 0.4 * np.sin(t_local * np.pi * 4)
    else:
        return T_critical + ((p - 0.6) / 0.4) * (4.0 - T_critical)


def run_experiment(L=8, total_steps=3000, seed=42, use_consensus=True, use_pivots=True):
    env = IsingEnvironment(L=L, seed=seed)
    agent = TUFEIsingAgent(L=L, seed=seed)
    consensus = ConsensusKnowledgeStore(min_observers=2)
    feasible_tracker = FeasibleSpaceTracker(window_size=20)
    phase_detector = PhaseTransitionDetector()
    observers = {
        'magnitude': ObserverMagnetization(),
        'correlation': ObserverCorrelation(),
        'energy': ObserverEnergy(),
    }
    history = []
    n_viable = 0
    for step in range(total_steps):
        T = temperature_schedule(step, total_steps)
        env.step_count = step
        pre_state = env.get_state()
        obs_buckets = [obs.perceive(pre_state) for obs in observers.values()]
        confirmed = consensus.get_confirmed_pivots() if (use_consensus and use_pivots) else {}
        action_id = agent.choose_action(obs_buckets, confirmed)
        agent.execute(env, action_id)
        env.apply_thermal_noise(T)
        post_state = env.get_state()
        if use_pivots:
            store = consensus if use_consensus else ConsensusKnowledgeStore(min_observers=99)
            viable = agent.learn(pre_state, post_state, action_id, observers, store)
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
                'step': step, 'T': round(T, 3),
                'mag': round(pre_state['magnetization'], 4),
                'corr': round(pre_state.get('correlation', 0), 4),
                'eps': round(pre_state.get('energy_per_site', 0), 4),
                'confirmed': len(conf_now), 'volume': vol,
                'phase_transition': trans is not None,
                'viability_rate': round(n_viable / max(step+1, 1), 3),
            })
    stats = {
        'total_steps': total_steps,
        'viability_rate': round(n_viable / max(total_steps, 1), 4),
        'total_actions': agent.total_actions,
        'viable_actions': agent.viable_count,
        'n_private_pivots': len(agent.private_pivots),
        'n_confirmed': len(consensus.confirmed) if use_consensus else 0,
        'feasible_space': feasible_tracker.get_report(),
        'phase_transitions': phase_detector.get_report(),
    }
    return {'config': {'L': L, 'steps': total_steps, 'seed': seed,
                       'consensus': use_consensus, 'pivots': use_pivots},
            'history': history, 'stats': stats}


def run_full_validation(n_seeds=5, total_steps=3000):
    print("=" * 70)
    print("TUF-E v0.2 Validation: Ising Domain v14.1 (Agent-Driven)")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Design: 3 conditions x {n_seeds} seeds x {total_steps} steps")
    print(f"Lattice: 8x8, Tc ~ 2.269, Agent = primary dynamics driver")
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
            r = run_experiment(L=8, total_steps=total_steps, seed=sv,
                             use_consensus=params['use_consensus'],
                             use_pivots=params['use_pivots'])
            seed_results.append(r)
            st = r['stats']
            print(f"  Seed {s}: VR={st['viability_rate']:.3f} | "
                  f"Pivots={st['n_confirmed']} | "
                  f"PT={st['phase_transitions']['n_transitions']} | "
                  f"Vol={st['feasible_space']['current_volume']}")
        all_results[cond_name] = seed_results
    summary = {}
    for cn, results in all_results.items():
        vrs = [r['stats']['viability_rate'] for r in results]
        vols = [r['stats']['feasible_space']['current_volume'] for r in results]
        peaks = [r['stats']['feasible_space']['peak_volume'] for r in results]
        npts = [r['stats']['phase_transitions']['n_transitions'] for r in results]
        nconfs = [r['stats']['n_confirmed'] for r in results]
        summary[cn] = {
            'viability_rate': {'mean': round(float(np.mean(vrs)), 4),
                              'std': round(float(np.std(vrs)), 4)},
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
        fv = d['feasible_volume']
        pt = d['phase_transitions']
        print(f"\n  {cn}:")
        print(f"    Viability Rate: {vr['mean']:.3f} +/- {vr['std']:.3f}")
        print(f"    Feasible Volume: {fv['mean_current']:.1f} (peak: {fv['mean_peak']:.1f})")
        print(f"    Phase Transitions: {pt['mean_count']:.1f} avg ({pt['total']} total)")
        print(f"    Confirmed Pivots: {d['confirmed_pivots']['mean']:.1f}")
    print("\n" + "=" * 70)
    print("CROSS-CONDITION ANALYSIS")
    print("=" * 70)
    if 'A_pivot_consensus' in summary and 'C_no_pivot_baseline' in summary:
        a_vr = summary['A_pivot_consensus']['viability_rate']['mean']
        c_vr = summary['C_no_pivot_baseline']['viability_rate']['mean']
        a_vol = summary['A_pivot_consensus']['feasible_volume']['mean_current']
        c_vol = summary['C_no_pivot_baseline']['feasible_volume']['mean_current']
        a_pt = summary['A_pivot_consensus']['phase_transitions']['mean_count']
        print(f"\n  A vs C (pivot+consensus vs baseline):")
        print(f"    VR difference: {a_vr - c_vr:+.3f}")
        print(f"    Volume difference: {a_vol - c_vol:+.1f}")
        print(f"    Phase transitions: {a_pt:.1f} vs 0")
        if a_vr > c_vr + 0.01:
            print(f"    Verdict: PIVOT+CONSENSUS IMPROVES VIABILITY")
        elif a_vr > c_vr - 0.01:
            print(f"    Verdict: COMPARABLE (pivot adds structure without cost)")
        else:
            print(f"    Verdict: BASELINE OUTPERFORMS (investigate why)")
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
            print(f"    Verdict: CONSENSUS NEUTRAL FOR VIABILITY")
        else:
            print(f"    Verdict: NO-CONSENSUS OUTPERFORMS")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, 'v14_ising_validation_results.json')
    output = {
        'experiment': 'TUF-E v0.2 Validation: Ising Domain v14.1',
        'timestamp': datetime.now().isoformat(),
        'design': 'Agent-driven dynamics, 8x8 lattice, 3-phase temperature protocol',
        'conditions': summary,
        'gc_mapping': {
            'GC-1': 'Site-level flip actions (129 options)',
            'GC-2': '3 observers: magnetization, correlation, energy',
            'GC-3': 'Pivot decay(0.003) + thermal noise erosion',
            'GC-4': 'Observe->Act->Noise->Observe = work cycle',
            'GC-5': 'Viability = non-trivial structure maintenance',
            'GC-6': 'Ising Hamiltonian immutable',
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")
    return summary


if __name__ == '__main__':
    run_full_validation(n_seeds=5, total_steps=3000)
