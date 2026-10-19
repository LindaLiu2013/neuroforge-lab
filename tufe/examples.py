"""
TUF-E Prototype 1.0: Example Domains

Demonstrates how to implement DomainSpec for different domains.
Each example maps GC-1~GC-6 to domain-specific constructs.

Included:
  - ArithmeticDomain: discrete exact causality (high consensability)
  - IsingDomain: continuous probabilistic causality (low consensability)
  - NKDomain: combinatorial epistatic causality (medium consensability)
"""

import numpy as np
from typing import Any

from .domain import DomainSpec, ObserverSpec, ActionResult, DissipationConfig


# ============================================================
# ARITHMETIC DOMAIN (reference implementation)
# ============================================================
class _ArithObsMagnitude(ObserverSpec):
    """Sees numbers through magnitude lens."""
    def __init__(self):
        super().__init__(name='magnitude', description='Absolute value magnitude + sign', n_buckets=16)

    def discretize(self, raw_observation: Any) -> str:
        if isinstance(raw_observation, (list, tuple)):
            val = raw_observation[0] if raw_observation else 0
        else:
            val = raw_observation or 0
        av = abs(val)
        sign_bit = 0 if val >= 0 else 1
        thresholds = [3, 6, 10, 15, 25, 40, 60, 100]
        level = sum(1 for t in thresholds if av >= t)
        idx = min(level + sign_bit * 8, 15)
        return f'b{idx:02d}'


class _ArithObsSymbolic(ObserverSpec):
    """Sees numbers through symbolic properties (parity, primality)."""
    def __init__(self):
        super().__init__(name='symbolic', description='Parity + sign + complexity', n_buckets=16)

    def discretize(self, raw_observation: Any) -> str:
        if isinstance(raw_observation, (list, tuple)):
            val = raw_observation[0] if raw_observation else 0
        else:
            val = raw_observation or 0
        parity = int(val) % 2
        sign_bit = 0 if val >= 0 else 1
        av = abs(val)
        size = 0 if av < 5 else (1 if av < 15 else (2 if av < 50 else 3))
        idx = min(size * 4 + sign_bit * 2 + parity, 15)
        return f'b{idx:02d}'


class _ArithObsTemporal(ObserverSpec):
    """Sees numbers through temporal clarity lens."""
    def __init__(self, rng=None):
        super().__init__(name='temporal', description='Magnitude + noise clarity', n_buckets=16)
        self._rng = rng or np.random.default_rng(42)

    def discretize(self, raw_observation: Any) -> str:
        if isinstance(raw_observation, (list, tuple)):
            val = raw_observation[0] if raw_observation else 0
        else:
            val = raw_observation or 0
        av = abs(val)
        noise_bit = int(self._rng.random() > 0.5)
        thresholds = [3, 6, 10, 15, 25, 40, 60, 100]
        level = sum(1 for t in thresholds if av >= t)
        idx = min(level * 2 + noise_bit, 15)
        return f'b{idx:02d}'


class ArithmeticDomain(DomainSpec):
    """
    Arithmetic operations domain.

    GC-1: 8 operations (add/sub/mul/div/neg/abs/sqr/id) with free composition
    GC-2: 3 observers (magnitude/symbolic/temporal) with independent discretization
    GC-3: Pivot decay + feedback delay creates temporal pressure
    GC-4: observe operands → choose operation → see result
    GC-5: Viability = correct answer (maintains problem-solving capability)
    GC-6: Operation definitions are immutable
    """

    OPS = {
        0: ('add', lambda a, b: a + b),
        1: ('sub', lambda a, b: a - b),
        2: ('mul', lambda a, b: a * b),
        3: ('div', lambda a, b: a / b if b != 0 else float('inf')),
        4: ('neg', lambda a, b: -a),
        5: ('abs', lambda a, b: abs(a)),
        6: ('sqr', lambda a, b: a * a),
        7: ('id', lambda a, b: a),
    }

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self._current_task = None
        self._obs_temporal = _ArithObsTemporal(rng=self.rng)

    def name(self) -> str:
        return "Arithmetic"

    def observers(self) -> list[ObserverSpec]:
        return [_ArithObsMagnitude(), _ArithObsSymbolic(), self._obs_temporal]

    def observe(self) -> dict[str, Any]:
        if self._current_task is None:
            self._generate_task()
        return {
            'magnitude': self._current_task['operands'],
            'symbolic': self._current_task['operands'],
            'temporal': self._current_task['operands'],
        }

    def action_space_size(self) -> int:
        return 8

    def act(self, action_id: int) -> ActionResult:
        if self._current_task is None:
            self._generate_task()

        task = self._current_task
        operands = list(task['operands'])

        try:
            _, fn = self.OPS.get(action_id, self.OPS[7])
            if len(operands) >= 2:
                result = fn(operands[0], operands[1])
            else:
                result = fn(operands[0], 0)

            diff = abs(result - task['target'])
            correct = diff < 0.001
        except Exception:
            result = None
            correct = False

        self._generate_task()

        return ActionResult(
            success=correct,
            post_state={'result': result, 'operands': self._current_task['operands'], 'correct': correct},
            metadata={'correct': correct, 'target': task['target']}
        )

    def assess_viability(self, state: dict[str, Any]) -> bool:
        return state.get('correct', False)

    def apply_dissipation(self) -> None:
        pass  # Arithmetic domain has no environmental dissipation

    def meta_rules(self) -> dict[str, Any]:
        return {'immutable_operations': list(self.OPS.keys())}

    def _generate_task(self):
        a = int(self.rng.integers(1, 20))
        b = int(self.rng.integers(1, 20))
        op_id = int(self.rng.choice([0, 1, 2]))
        _, fn = self.OPS[op_id]
        target = fn(a, b)
        self._current_task = {'operands': [a, b], 'target': target, 'solution_op': op_id}


# ============================================================
# ISING DOMAIN
# ============================================================
class IsingDomain(DomainSpec):
    """
    2D Ising model domain.

    GC-1: Flip any site on L×L lattice (2L²+1 actions)
    GC-2: 3 observers (magnetization/correlation/energy)
    GC-3: Thermal noise erodes agent's work
    GC-4: observe→flip→thermal_noise→observe
    GC-5: Viability = non-trivial structure (not fully ordered/disordered)
    GC-6: Ising Hamiltonian immutable
    """

    def __init__(self, L=8, seed=42):
        self.L = L
        self.N = L * L
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(L, L)).astype(np.int8)
        self.J = 1.0
        self.temperature = 2.269

    def name(self) -> str:
        return f"Ising-{self.L}x{self.L}"

    def observers(self) -> list[ObserverSpec]:
        return [_IsingObsMag(), _IsingObsCorr(), _IsingObsEnergy()]

    def observe(self) -> dict[str, Any]:
        mag = float(np.sum(self.spins)) / self.N
        corr = self._correlation()
        energy = self._energy() / self.N
        return {'magnetization': mag, 'correlation': corr, 'energy': energy}

    def action_space_size(self) -> int:
        return self.N * 2 + 1

    def act(self, action_id: int) -> ActionResult:
        if action_id < self.N * 2:
            site = action_id // 2
            direction = action_id % 2
            i, j = site // self.L, site % self.L
            self.spins[i, j] = 1 if direction == 0 else -1

        mag = float(np.sum(self.spins)) / self.N
        corr = self._correlation()
        energy = self._energy() / self.N
        return ActionResult(
            success=True,
            post_state={'magnetization': mag, 'correlation': corr, 'energy_per_site': energy}
        )

    def assess_viability(self, state: dict[str, Any]) -> bool:
        am = abs(state.get('magnetization', 0))
        corr = state.get('correlation', 0.5)
        eps = state.get('energy_per_site', -1.0)
        return (0.1 <= am <= 0.8) and (corr >= 0.4) and (eps > -1.8)

    def apply_dissipation(self) -> None:
        T = self.temperature
        if T <= 0:
            return
        for _ in range(self.N // 2):
            i = int(self.rng.integers(0, self.L))
            j = int(self.rng.integers(0, self.L))
            s = self.spins[i, j]
            neighbors = (self.spins[i, (j+1)%self.L] + self.spins[i, (j-1)%self.L] +
                        self.spins[(i+1)%self.L, j] + self.spins[(i-1)%self.L, j])
            delta_e = 2.0 * self.J * s * neighbors
            if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / T):
                self.spins[i, j] *= -1

    def meta_rules(self) -> dict[str, Any]:
        return {'hamiltonian': 'H = -J Σ<ij> σᵢσⱼ', 'J': self.J, 'lattice': f'{self.L}x{self.L}'}

    def _correlation(self) -> float:
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

    def _energy(self) -> float:
        E = 0.0
        for i in range(self.L):
            for j in range(self.L):
                s = self.spins[i, j]
                right = self.spins[i, (j + 1) % self.L]
                down = self.spins[(i + 1) % self.L, j]
                E -= self.J * s * (right + down)
        return E


class _IsingObsMag(ObserverSpec):
    def __init__(self):
        super().__init__(name='magnetization', description='Global order parameter')
    def discretize(self, raw_observation: Any) -> str:
        av = abs(raw_observation) if raw_observation is not None else 0
        sign_bit = 0 if (raw_observation or 0) >= 0 else 1
        thresholds = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9]
        level = sum(1 for t in thresholds if av >= t)
        idx = min(level + sign_bit * 8, 15)
        return f'b{idx:02d}'


class _IsingObsCorr(ObserverSpec):
    def __init__(self):
        super().__init__(name='correlation', description='Local spatial structure')
    def discretize(self, raw_observation: Any) -> str:
        v = max(0.0, min(1.0, (raw_observation - 0.3) / 0.7)) if raw_observation is not None else 0.5
        idx = min(int(v * 16), 15)
        return f'b{idx:02d}'


class _IsingObsEnergy(ObserverSpec):
    def __init__(self):
        super().__init__(name='energy', description='Thermodynamic stability')
    def discretize(self, raw_observation: Any) -> str:
        v = max(0.0, min(1.0, (raw_observation + 2.0) / 2.0)) if raw_observation is not None else 0.5
        idx = min(int(v * 16), 15)
        return f'b{idx:02d}'


# ============================================================
# NK BOOLEAN NETWORK DOMAIN
# ============================================================
class NKDomain(DomainSpec):
    """
    Kauffman NK Boolean network domain.

    GC-1: Toggle any of N genes (N+1 actions including do-nothing)
    GC-2: 3 observers (state_pattern/fitness/dynamics)
    GC-3: Environmental perturbation flips random genes
    GC-4: observe→toggle→boolean_update→perturb→observe
    GC-5: Viability = fitness in survivable range [0.3, 0.7]
    GC-6: NK landscape topology + update rules fixed
    """

    def __init__(self, N=12, K=3, seed=42):
        self.N_genes = N
        self.K = min(K, N - 1)
        self.rng = np.random.default_rng(seed)

        self.connections = []
        for i in range(N):
            possible = [j for j in range(N) if j != i]
            self.rng.shuffle(possible)
            self.connections.append(sorted(possible[:self.K]))

        self.truth_tables = [self.rng.integers(0, 2, size=2**self.K).astype(np.int8) for _ in range(N)]
        self.fitness_tables = [self.rng.random(size=2**self.K) for _ in range(N)]
        self.state = self.rng.integers(0, 2, size=N).astype(np.int8)

    def name(self) -> str:
        return f"NK-N{self.N_genes}-K{self.K}"

    def observers(self) -> list[ObserverSpec]:
        return [_NKObsState(), _NKObsFitness(), _NKObsDynamics()]

    def observe(self) -> dict[str, Any]:
        f = self._fitness()
        af = float(np.sum(self.state)) / self.N_genes
        ad = self._attractor_distance()
        return {'state_pattern': af, 'fitness': f, 'dynamics': ad}

    def action_space_size(self) -> int:
        return self.N_genes + 1

    def act(self, action_id: int) -> ActionResult:
        if action_id < self.N_genes:
            self.state[action_id] = 1 - self.state[action_id]

        # Boolean update
        new_state = np.zeros(self.N_genes, dtype=np.int8)
        for i in range(self.N_genes):
            inputs = self.connections[i]
            idx = sum((1 << bit) for bit, g in enumerate(inputs) if self.state[g])
            new_state[i] = self.truth_tables[i][idx]
        self.state = new_state

        f = self._fitness()
        af = float(np.sum(self.state)) / self.N_genes
        return ActionResult(success=True, post_state={'fitness': f, 'active_fraction': af})

    def assess_viability(self, state: dict[str, Any]) -> bool:
        f = state.get('fitness', 0.5)
        return 0.3 <= f <= 0.7

    def apply_dissipation(self) -> None:
        g = int(self.rng.integers(0, self.N_genes))
        self.state[g] = 1 - self.state[g]

    def meta_rules(self) -> dict[str, Any]:
        return {'N': self.N_genes, 'K': self.K, 'update_rule': 'synchronous_boolean'}

    def _fitness(self) -> float:
        total = 0.0
        for i in range(self.N_genes):
            inputs = self.connections[i]
            idx = sum((1 << bit) for bit, g in enumerate(inputs) if self.state[g])
            total += self.fitness_tables[i][idx]
        return total / self.N_genes

    def _attractor_distance(self) -> float:
        saved = self.state.copy()
        changes = 0
        for _ in range(5):
            old = self.state.copy()
            new_state = np.zeros(self.N_genes, dtype=np.int8)
            for i in range(self.N_genes):
                inputs = self.connections[i]
                idx = sum((1 << bit) for bit, g in enumerate(inputs) if self.state[g])
                new_state[i] = self.truth_tables[i][idx]
            self.state = new_state
            changes += np.sum(old != self.state)
        self.state = saved
        return changes / (5 * self.N_genes)


class _NKObsState(ObserverSpec):
    def __init__(self):
        super().__init__(name='state_pattern', description='Gene activity pattern')
    def discretize(self, raw_observation: Any) -> str:
        v = max(0.0, min(1.0, raw_observation)) if raw_observation is not None else 0.5
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]
        level = sum(1 for t in thresholds if v >= t)
        idx = min(level * 2, 15)
        return f'b{idx:02d}'


class _NKObsFitness(ObserverSpec):
    def __init__(self):
        super().__init__(name='fitness', description='Fitness landscape value')
    def discretize(self, raw_observation: Any) -> str:
        v = max(0.0, min(1.0, raw_observation)) if raw_observation is not None else 0.5
        idx = min(int(v * 16), 15)
        return f'b{idx:02d}'


class _NKObsDynamics(ObserverSpec):
    def __init__(self):
        super().__init__(name='dynamics', description='Attractor proximity')
    def discretize(self, raw_observation: Any) -> str:
        v = max(0.0, min(1.0, raw_observation)) if raw_observation is not None else 0.5
        idx = min(int(v * 16), 15)
        return f'b{idx:02d}'


# ============================================================
# REGEX DOMAIN — real-world learning target
# ============================================================
import re

class _RegexObsPattern(ObserverSpec):
    """Sees the regex pattern's structural composition."""
    def __init__(self):
        super().__init__(name='pattern_structure', description='Regex syntax composition', n_buckets=16)

    def discretize(self, raw_observation: Any) -> str:
        if not isinstance(raw_observation, str):
            return 'b00'
        features = 0
        if '.' in raw_observation: features |= 1
        if '*' in raw_observation or '+' in raw_observation: features |= 2
        if '?' in raw_observation: features |= 4
        if '[' in raw_observation: features |= 8
        if '(' in raw_observation: features |= 16
        if '^' in raw_observation or '$' in raw_observation: features |= 32
        if '\\' in raw_observation: features |= 64
        if '|' in raw_observation: features |= 128
        idx = min(features % 16, 15)
        return f'b{idx:02d}'


class _RegexObsMatch(ObserverSpec):
    """Sees match results against test strings."""
    def __init__(self):
        super().__init__(name='match_result', description='Match success rate + coverage', n_buckets=16)

    def discretize(self, raw_observation: Any) -> str:
        if isinstance(raw_observation, dict):
            rate = raw_observation.get('match_rate', 0.0)
        else:
            rate = float(raw_observation) if raw_observation is not None else 0.0
        idx = min(int(rate * 16), 15)
        return f'b{idx:02d}'


class _RegexObsComplexity(ObserverSpec):
    """Sees pattern complexity (length + char diversity)."""
    def __init__(self):
        super().__init__(name='complexity', description='Pattern length and character diversity', n_buckets=16)

    def discretize(self, raw_observation: Any) -> str:
        if not isinstance(raw_observation, str):
            return 'b00'
        length_score = min(len(raw_observation) / 20.0, 1.0)
        unique_chars = len(set(raw_observation))
        diversity_score = min(unique_chars / 10.0, 1.0)
        combined = (length_score + diversity_score) / 2.0
        idx = min(int(combined * 16), 15)
        return f'b{idx:02d}'


class RegexDomain(DomainSpec):
    """
    Regular expression learning domain.

    The agent learns to construct regex patterns that match target descriptions.
    GC-1: 12 regex construction actions (append char/class/quantifier/anchor/group/etc.)
    GC-2: 3 observers (pattern_structure, match_result, complexity) with independent discretization
    GC-3: Pivot decay; unused regex knowledge fades
    GC-4: observe target → build pattern → test against strings → see results
    GC-5: Viability = pattern matches >=60% of positive examples AND rejects >=60% of negatives
    GC-6: Python regex engine semantics are immutable
    """

    ACTIONS = [
        ('append_literal', lambda p, _: p + 'a'),
        ('append_digit', lambda p, _: p + r'\d'),
        ('append_word', lambda p, _: p + r'\w'),
        ('append_any', lambda p, _: p + '.'),
        ('append_star', lambda p, _: p + '*' if p else p),
        ('append_plus', lambda p, _: p + '+' if p else p),
        ('append_question', lambda p, _: p + '?' if p else p),
        ('append_anchor_start', lambda p, _: '^' + p),
        ('append_anchor_end', lambda p, _: p + '$'),
        ('append_char_class', lambda p, _: p + '[a-z]'),
        ('append_group', lambda p, _: '(' + p + ')' if p else p),
        ('reset', lambda p, _: ''),
    ]

    TASKS = [
        {'desc': 'starts with digit', 'pos': ['1abc', '9xyz', '3test'], 'neg': ['abc', 'xyz', 'test']},
        {'desc': 'contains number', 'pos': ['abc123', 'test5', 'x9y'], 'neg': ['abc', 'xyz', 'hello']},
        {'desc': 'ends with ing', 'pos': ['running', 'testing', 'going'], 'neg': ['run', 'test', 'gone']},
        {'desc': 'has uppercase', 'pos': ['Hello', 'ABC', 'aBc'], 'neg': ['hello', 'abc', 'xyz']},
        {'desc': 'is all digits', 'pos': ['123', '42', '0'], 'neg': ['12a', 'abc', '1.5']},
        {'desc': 'has special char', 'pos': ['a@b', 'x!y', 'test#'], 'neg': ['abc', 'xyz', 'hello']},
        {'desc': 'two words', 'pos': ['hello world', 'foo bar', 'a b'], 'neg': ['hello', 'helloworld', '']},
        {'desc': 'email-like', 'pos': ['a@b.c', 'x@y.z', 'me@co.uk'], 'neg': ['abc', '@bc', 'a@c']},
    ]

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self._current_pattern = ''
        self._current_task_idx = 0
        self._cycles_on_task = 0
        self._task_duration = 200
        self._task_order = list(range(len(self.TASKS)))
        self.rng.shuffle(self._task_order)

    def name(self) -> str:
        return "Regex"

    def observers(self) -> list[ObserverSpec]:
        return [_RegexObsPattern(), _RegexObsMatch(), _RegexObsComplexity()]

    def observe(self) -> dict[str, Any]:
        task = self.TASKS[self._task_order[self._current_task_idx]]
        match_info = self._evaluate(self._current_pattern, task)
        return {
            'pattern_structure': self._current_pattern,
            'match_result': match_info,
            'complexity': self._current_pattern,
        }

    def action_space_size(self) -> int:
        return len(self.ACTIONS)

    def act(self, action_id: int) -> ActionResult:
        name, fn = self.ACTIONS[action_id % len(self.ACTIONS)]
        try:
            new_pattern = fn(self._current_pattern, None)
            if len(new_pattern) > 30:
                new_pattern = self._current_pattern
            re.compile(new_pattern)
            self._current_pattern = new_pattern
        except re.error:
            pass

        task = self.TASKS[self._task_order[self._current_task_idx]]
        match_info = self._evaluate(self._current_pattern, task)
        viable = match_info['match_rate'] >= 0.6 and match_info['reject_rate'] >= 0.6

        self._cycles_on_task += 1
        if self._cycles_on_task >= self._task_duration:
            self._current_task_idx = (self._current_task_idx + 1) % len(self._task_order)
            self._cycles_on_task = 0
            self._current_pattern = ''

        return ActionResult(
            success=viable,
            post_state={
                'pattern': self._current_pattern,
                'match_rate': match_info['match_rate'],
                'reject_rate': match_info['reject_rate'],
                'viable': viable,
            },
            metadata={'task': task['desc']}
        )

    def assess_viability(self, state: dict[str, Any]) -> bool:
        return state.get('viable', False)

    def apply_dissipation(self) -> None:
        if self.rng.random() < 0.05 and self._current_pattern:
            self._current_pattern = self._current_pattern[:-1]

    def meta_rules(self) -> dict[str, Any]:
        return {'regex_engine': 'python-re', 'max_pattern_length': 30}

    def _evaluate(self, pattern: str, task: dict) -> dict:
        if not pattern:
            return {'match_rate': 0.0, 'reject_rate': 1.0}
        try:
            compiled = re.compile(pattern)
            pos_hits = sum(1 for s in task['pos'] if compiled.search(s))
            neg_hits = sum(1 for s in task['neg'] if compiled.search(s))
            match_rate = pos_hits / max(len(task['pos']), 1)
            reject_rate = 1.0 - neg_hits / max(len(task['neg']), 1)
            return {'match_rate': round(match_rate, 3), 'reject_rate': round(reject_rate, 3)}
        except re.error:
            return {'match_rate': 0.0, 'reject_rate': 1.0}
