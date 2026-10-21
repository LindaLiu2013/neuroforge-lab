"""v1.9 (Q1): ACTIVE INFORMATION ACQUISITION vs RE-ADDRESSING — circle->spiral
boundary. Pre-registered in poc/preregister/v19_active_info.yaml under
PROTOCOL_autonomous_run.md. This is a FALSIFICATION-FIRST test: the expected
result is NEGATIVE and the negative is the main product.

Drift anchor (recite before reading results): NO reward, NO gradient, NO
cross-step credit assignment, NO scoring. Viability (correct/incorrect) is the
sole acceptor. rho = 0 (no environmental rescue).

--------------------------------------------------------------------------
THE QUESTION
--------------------------------------------------------------------------
v1.8 showed the engine can unlock a task by RE-ADDRESSING information that was
ALREADY in the observation stream (zero new information; a "circle"). Here we
build a domain where the needed information is NOT in the passive stream at all:
it is hidden behind a PROBE action that is LOCALLY NON-VIABLE (probing fails the
current cycle) and whose only value is revealing future information.

A viability-immediate engine with no reward and no credit assignment has, in
principle, no way to prefer an action that is locally non-viable for a future
gain. Prediction (H-LUCK-ONLY): the engine will NOT discover a probe-first
policy; it will exploit revealed states when it stumbles into them by random
exploration, but never systematically probe to CREATE them. If that holds, it
localizes SC-3 and marks the circle->spiral boundary.

Re-addressing (v1.8's operator, extended here over the USELESS token window)
provably cannot help: the hidden bit h is not a chunking problem, it is absent
from the token stream. If extending the window nonetheless breaks the guess
ceiling, my "re-addressing != acquiring absent info" claim is FALSIFIED (H-LEAK).

Usage: python poc/v19_active_info.py <arm> <seed> <n_cycles>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.domain import (DomainSpec, ObserverSpec, ActionResult,
                         DissipationConfig, ConsensusConfig)
from poc.v18_third_axis import MetaCompressionOperator

N_ACTIONS = 3          # 0, 1, PROBE(=2)
PROBE = 2
TASK_DURATION = 20     # D: hidden bit refreshes every D cycles
CHANCE_FLOOR = 0.5     # guessing between {0,1}


class _ObsToken(ObserverSpec):
    def __init__(self):
        super().__init__(name='token_id', description='useless random token', n_buckets=16)
    def discretize(self, raw):
        return f'b{int(raw) % 16:02d}' if raw is not None else 'b00'


class _ObsRevealed(ObserverSpec):
    def __init__(self):
        super().__init__(name='revealed', description='is hidden bit currently visible', n_buckets=16)
    def discretize(self, raw):
        return 'b01' if raw else 'b00'


class _ObsHint(ObserverSpec):
    def __init__(self):
        super().__init__(name='hint', description='visible hidden-bit value or mask', n_buckets=16)
    def discretize(self, raw):
        return 'b00' if raw is None else f'b{int(raw):02d}'


class ActiveInfoDomain(DomainSpec):
    """Hidden bit h refreshed each task; revealed only by a locally-non-viable
    PROBE. Passive token stream carries zero information about h.

    oracle_reveal=True => h visible from cycle 0 (designer supplies the info:
    EXTERNAL door ceiling, NOT endogenous acquisition).
    """

    def __init__(self, seed=42, oracle_reveal=False):
        self.rng = np.random.default_rng(seed)
        self.oracle_reveal = oracle_reveal
        self._new_task()
        self.cycles_in_task = 0
        self.task_index = 0
        # probe-timing diagnostics
        self.probe_positions = []      # cycles_in_task at each PROBE
        self.revealed_cycles = 0
        self.total_cycles = 0
        # token history for the (useless) re-addressing window
        self._hist = [0, 0, 0]

    def _new_task(self):
        self.h = int(self.rng.integers(0, 2))
        self.revealed = self.oracle_reveal
        self.cycles_in_task = 0
        self.cur_token = int(self.rng.integers(0, 4))

    def name(self):
        return 'ActiveInfo' + ('-oracle' if self.oracle_reveal else '')

    def observers(self):
        return [_ObsToken(), _ObsRevealed(), _ObsHint()]

    def observe(self):
        return {
            'token_id': self.cur_token,
            'revealed': 1 if self.revealed else 0,
            'hint': self.h if self.revealed else None,
            # window fields (useless token history) for the address key fn:
            'w_hist': list(self._hist),
            'w_revealed': 1 if self.revealed else 0,
            'w_hint': self.h if self.revealed else -1,
        }

    def action_space_size(self):
        return N_ACTIONS

    def act(self, action_id):
        a = action_id % N_ACTIONS
        self.total_cycles += 1
        if a == PROBE:
            self.probe_positions.append(self.cycles_in_task)
            viable = False                    # locally NON-viable
            self.revealed = True              # but reveals h for rest of task
        else:
            viable = (a == self.h)
        if self.revealed:
            self.revealed_cycles += 1

        # advance task clock
        self.cycles_in_task += 1
        if self.cycles_in_task >= TASK_DURATION:
            self.task_index += 1
            self._new_task()
        else:
            self._hist = [self.cur_token] + self._hist[:2]
            self.cur_token = int(self.rng.integers(0, 4))

        return ActionResult(
            success=viable,
            post_state={'viable': viable, 'revealed': self.revealed},
            metadata={'h': self.h, 'action': a},
        )

    def assess_viability(self, state):
        return bool(state.get('viable', False))

    def apply_dissipation(self):
        pass  # rho = 0

    def dissipation_config(self):
        return DissipationConfig(pivot_decay_rate=0.003)

    def consensus_config(self):
        return ConsensusConfig(min_observers=2, strategy='standard')

    def meta_rules(self):
        return {'h_rule': 'environment-only', 'PROBE_cost': 'locally non-viable',
                'oracle': self.oracle_reveal, 'note': 'operator cannot read h'}


def make_key_fn(window_holder):
    """Key always carries the USEFUL observable (revealed, hint); the window w
    extends only over the USELESS token history — so extending it cannot add
    information about h (that is the point)."""
    def key_fn(raw):
        w = window_holder['w']
        toks = ''.join(str(x) for x in raw['w_hist'][:max(0, w)])
        return f"r{raw['w_revealed']}_h{raw['w_hint']}_t{toks}"
    return key_fn


ARMS = {
    'guess_frozen':  dict(oracle=False, enabled=False, window0=1),
    'operator_on':   dict(oracle=False, enabled=True,  window0=1),
    'oracle_reveal': dict(oracle=True,  enabled=False, window0=1),
}


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    cfg = ARMS[arm]

    domain = ActiveInfoDomain(seed=seed, oracle_reveal=cfg['oracle'])
    window_holder = {'w': cfg['window0']}
    agent = TUFEAgent(domain, seed=seed, exact_addressing=True,
                      address_key_fn=make_key_fn(window_holder))
    op = MetaCompressionOperator(window_holder, enabled=cfg['enabled'], window0=cfg['window0'])

    domain.initial_perturbation()
    viable_cycles = 0
    t0 = time.time()
    for c in range(n_cycles):
        summary = agent.run_cycle()
        viable_cycles += 1 if summary['viable'] else 0
        op.maybe_mint(agent, c + 1)
    seconds = time.time() - t0

    vr = viable_cycles / max(n_cycles, 1)
    # OC-9: excess over the guess floor (rho=0 => no rescue; floor is guessing 0.5)
    excess = vr - CHANCE_FLOOR
    # probe-policy signature: mean cycles-into-task at which PROBE fired
    pp = domain.probe_positions
    mean_probe_pos = (sum(pp) / len(pp)) if pp else None
    probe_rate = len(pp) / max(n_cycles, 1)
    # a learned probe-first policy => mean_probe_pos small (< D/3); random => ~D/2
    revealed_frac = domain.revealed_cycles / max(n_cycles, 1)

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles,
        'VR_agent': round(vr, 4),
        'chance_floor': CHANCE_FLOOR,
        'OC9_excess_over_guess': round(excess, 4),
        'n_probes': len(pp),
        'probe_rate': round(probe_rate, 4),
        'mean_probe_pos_in_task': round(mean_probe_pos, 3) if mean_probe_pos is not None else None,
        'task_duration_D': TASK_DURATION,
        'revealed_frac': round(revealed_frac, 4),
        'n_mint_events': len(op.mint_events),
        'final_window': window_holder['w'],
        'exact_fires': agent.exact_fires,
        'fire_viable_rate': round(agent.exact_fire_viable / max(agent.exact_fires, 1), 4),
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
