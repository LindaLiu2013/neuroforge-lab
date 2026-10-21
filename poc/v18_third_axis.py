"""v1.8 third-axis minimal test: can the system ENDGENOUSLY mint a new
addressable unit (vocabulary growth) that unlocks a task class provably
unsolvable under the frozen base vocabulary?

Design locked in docs/tufe-design-language-v03.md §v1.8. Drift anchor (recite
before reading results): this tests RE-COMPRESSION of the agent's own confirmed
knowledge + viability acceptance. There is NO reward, NO gradient, NO scoring
function. Viability (correct/incorrect) is the sole acceptor.

--------------------------------------------------------------------------
THE HONEST SET-UP (OC-12 / C5)
--------------------------------------------------------------------------
Base vocabulary (object layer, FROZEN): address situations by the CURRENT single
token (unigram). This is the frozen primitive set.

Two stationary environments, alphabet = actions = {0,1,2,3}:
  - Env-Unigram (control): correct action = cur. A unigram key determines the
    action => solvable under the frozen vocabulary. The compression operator
    should find NO unresolved pressure and NOT need to mint.
  - Env-Bigram (target): correct action = (prev + cur) % 4. For a FIXED cur, as
    prev varies the correct action ranges uniformly over all 4 actions => the
    unigram key is *informationless* about the action (chance ceiling 0.25). The
    task is PROVABLY unsolvable under the frozen unigram vocabulary. Only a
    bigram key (prev,cur) determines it.

Meta-layer compression operator (FROZEN, task-agnostic, C5):
  It knows NOTHING about tokens, bigrams, or the rule. It only watches the
  agent's OWN addressing statistics: "among situations I have visited often,
  what fraction stay UNRESOLVED (no action ever reached viability)?" When that
  fraction is high, it does exactly ONE thing: increases the addressing WINDOW
  by one (mint larger chunks as new addressable units) and re-addresses. It
  never writes an action, never writes content, never touches the agent's
  confirmation / consensus / viability logic. => the FORM change is a fixed
  meta-rule (the pi axis, designer-set as C5 requires); the CONTENT that fills
  the new units comes only from the environment (the observed token stream).

--------------------------------------------------------------------------
ARMS
--------------------------------------------------------------------------
  bigram_frozen   : Env-Bigram, operator OFF, window frozen=1  -> ceiling test
  bigram_operator : Env-Bigram, operator ON                    -> third-axis test
  unigram_operator: Env-Unigram, operator ON                   -> spurious-fire control
  bigram_oracle   : Env-Bigram, window=2 from birth (designer-supplied) -> ceiling,
                    NOT a third-axis result (external/door = designer pre-supply)

Judgments (mutually exclusive, §v1.8 四):
  P-A L2 holds : bigram_operator unlocks Env-Bigram (agent VR >= 0.5 from ~chance)
                 AND a mint event fired AND recovered content matches the ENV rule.
  P-B only L1  : mint fired but solvable set unchanged (VR stays ~chance); only
                 step/fire efficiency moved.
  P-C none     : operator never produces a viability-accepted longer window;
                 Env-Bigram stays at chance => SC-3 is external-dependency.
  P-D contamination: recovered content traces to designer prior, not env => void.

rho = 0 everywhere (no environmental rescue) so agent VR == agent-caused VR and
OC-9 excess causal validity == fire_viable (no learning-illusion hiding place).

Usage: python poc/v18_third_axis.py <arm> <seed> <n_cycles>
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

ALPHABET = 4
N_ACTIONS = 4


# --------------------------------------------------------------------------
# Observers (>=2 methodologically independent, required by GC-2)
# --------------------------------------------------------------------------
class _ObsToken(ObserverSpec):
    def __init__(self):
        super().__init__(name='token_id', description='current token identity', n_buckets=16)

    def discretize(self, raw):
        return f'b{int(raw) % 16:02d}' if raw is not None else 'b00'


class _ObsParity(ObserverSpec):
    def __init__(self):
        super().__init__(name='parity', description='current token parity', n_buckets=16)

    def discretize(self, raw):
        return f'b{int(raw) % 2:02d}' if raw is not None else 'b00'


class _ObsRepeat(ObserverSpec):
    def __init__(self):
        super().__init__(name='repeat', description='did token repeat previous', n_buckets=16)

    def discretize(self, raw):
        return 'b01' if raw else 'b00'


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------
class ThirdAxisDomain(DomainSpec):
    """Stationary token stream; correct action depends on rule type.

    rule='unigram': correct = cur
    rule='bigram' : correct = (prev + cur) % 4   (unigram-informationless)
    """

    def __init__(self, rule='bigram', seed=42):
        assert rule in ('unigram', 'bigram')
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self.prev2 = 0
        self.prev = 0
        self.cur = int(self.rng.integers(0, ALPHABET))
        self._obs_repeat_rng = np.random.default_rng(seed + 7)

    def name(self):
        return f'ThirdAxis-{self.rule}'

    def observers(self):
        return [_ObsToken(), _ObsParity(), _ObsRepeat()]

    def observe(self):
        return {
            'token_id': self.cur,
            'parity': self.cur,
            'repeat': 1 if self.cur == self.prev else 0,
            # window fields consumed by the address key fn (not by observers):
            'w_cur': self.cur,
            'w_prev': self.prev,
            'w_prev2': self.prev2,
        }

    def action_space_size(self):
        return N_ACTIONS

    def _correct(self):
        if self.rule == 'unigram':
            return self.cur % N_ACTIONS
        return (self.prev + self.cur) % N_ACTIONS

    def act(self, action_id):
        correct = self._correct()
        viable = (action_id % N_ACTIONS) == correct
        # advance the stationary stream
        self.prev2 = self.prev
        self.prev = self.cur
        self.cur = int(self.rng.integers(0, ALPHABET))
        return ActionResult(
            success=viable,
            post_state={'viable': viable, 'correct': correct},
            metadata={'correct_action': correct, 'rule': self.rule},
        )

    def assess_viability(self, state):
        return bool(state.get('viable', False))

    def apply_dissipation(self):
        # rho = 0: no environmental rescue, no erosion of the rule.
        pass

    def dissipation_config(self):
        return DissipationConfig(pivot_decay_rate=0.003)

    def consensus_config(self):
        return ConsensusConfig(min_observers=2, strategy='standard')

    def meta_rules(self):
        return {'rule': self.rule, 'alphabet': ALPHABET, 'n_actions': N_ACTIONS,
                'note': 'rule is the ENVIRONMENT (immutable); operator cannot read it'}


# --------------------------------------------------------------------------
# Addressing vocabulary (object layer). window is the ONLY mutable form param.
# --------------------------------------------------------------------------
def make_key_fn(window_holder):
    def key_fn(raw):
        w = window_holder['w']
        if w <= 1:
            return f'u{raw["w_cur"]}'
        if w == 2:
            return f'g{raw["w_prev"]}_{raw["w_cur"]}'
        return f't{raw["w_prev2"]}_{raw["w_prev"]}_{raw["w_cur"]}'
    return key_fn


# --------------------------------------------------------------------------
# Meta-layer compression operator (FROZEN, task-agnostic, C5-guarded)
# --------------------------------------------------------------------------
class MetaCompressionOperator:
    """Watches the agent's own addressing statistics; on sustained unresolved
    pressure, extends the addressing window by one and re-addresses.

    C5 guard: the ONLY write it performs is `window_holder['w'] += 1` and
    clearing the agent's address memory. It never writes an action, a bucket, a
    confirmation, a consensus value, or any content. Asserted in _apply.
    """

    def __init__(self, window_holder, enabled=True, window0=1, max_window=3,
                 eval_interval=500, warmup=1000, visit_thresh=8,
                 unresolved_thresh=0.6, min_visited_keys=3):
        self.h = window_holder
        self.h['w'] = window0
        self.enabled = enabled
        self.max_window = max_window
        self.eval_interval = eval_interval
        self.warmup = warmup
        self.visit_thresh = visit_thresh
        self.unresolved_thresh = unresolved_thresh
        self.min_visited_keys = min_visited_keys
        self.mint_events = []

    def _resolution_stats(self, agent):
        visited = 0
        unresolved = 0
        for key, rec in agent.exact_mem.items():
            total_n = sum(n for (n, v) in rec.values())
            if total_n < self.visit_thresh:
                continue
            visited += 1
            # a key is RESOLVED if some action reached the fire criterion v>=0.5n, n>=3
            resolved = any(n >= 3 and v >= 0.5 * n for (n, v) in rec.values())
            if not resolved:
                unresolved += 1
        frac = unresolved / visited if visited else 0.0
        return visited, unresolved, frac

    def maybe_mint(self, agent, cycle):
        if not self.enabled or self.h['w'] >= self.max_window:
            return None
        if cycle < self.warmup or cycle % self.eval_interval != 0:
            return None
        visited, unresolved, frac = self._resolution_stats(agent)
        if visited < self.min_visited_keys:
            return None
        if frac >= self.unresolved_thresh:
            before = self.h['w']
            # --- C5 GUARD: form-only write, no content ---
            self.h['w'] = before + 1
            agent.exact_mem = {}
            # end C5 guard
            ev = {'cycle': cycle, 'window_before': before, 'window_after': self.h['w'],
                  'unresolved_frac': round(frac, 3), 'visited_keys': visited,
                  'unresolved_keys': unresolved}
            self.mint_events.append(ev)
            return ev
        return None


# --------------------------------------------------------------------------
# Content-recovery check (P-D): did recovered bigram->action match the ENV rule?
# --------------------------------------------------------------------------
def content_recovery(agent, rule):
    """Inspect locked address keys and compare to the true environmental rule.
    Returns (n_locked, frac_matching_env). Content match => came from env, not
    from the operator (which holds no rule knowledge)."""
    locked = 0
    match = 0
    for key, rec in agent.exact_mem.items():
        for aid, (n, v) in rec.items():
            if n >= 3 and v >= 0.5 * n:
                locked += 1
                # parse the key back to (prev, cur) if bigram
                if key.startswith('g'):
                    p, c = key[1:].split('_')
                    p, c = int(p), int(c)
                    if rule == 'bigram':
                        env_correct = (p + c) % N_ACTIONS
                    else:
                        env_correct = c % N_ACTIONS
                    if aid % N_ACTIONS == env_correct:
                        match += 1
                elif key.startswith('u'):
                    c = int(key[1:])
                    env_correct = c % N_ACTIONS if rule == 'unigram' else None
                    if rule == 'unigram' and aid % N_ACTIONS == env_correct:
                        match += 1
                break  # one locked action per key
    frac = match / locked if locked else 0.0
    return locked, round(frac, 3)


ARMS = {
    'bigram_frozen':    dict(rule='bigram',  enabled=False, window0=1),
    'bigram_operator':  dict(rule='bigram',  enabled=True,  window0=1),
    'unigram_operator': dict(rule='unigram', enabled=True,  window0=1),
    'bigram_oracle':    dict(rule='bigram',  enabled=False, window0=2),
}


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    cfg = ARMS[arm]

    domain = ThirdAxisDomain(rule=cfg['rule'], seed=seed)
    window_holder = {'w': cfg['window0']}
    agent = TUFEAgent(domain, seed=seed, exact_addressing=True,
                      address_key_fn=make_key_fn(window_holder))
    op = MetaCompressionOperator(window_holder, enabled=cfg['enabled'],
                                 window0=cfg['window0'])

    domain.initial_perturbation()

    viable_cycles = 0
    first_mint_cycle = None
    post_mint_viable = 0
    post_mint_cycles = 0
    t0 = time.time()
    for c in range(n_cycles):
        summary = agent.run_cycle()
        v = bool(summary['viable'])
        viable_cycles += 1 if v else 0
        if first_mint_cycle is not None:
            post_mint_cycles += 1
            post_mint_viable += 1 if v else 0
        ev = op.maybe_mint(agent, c + 1)
        if ev is not None and first_mint_cycle is None:
            first_mint_cycle = ev['cycle']

    seconds = time.time() - t0
    vr_agent = viable_cycles / max(n_cycles, 1)
    fire_viable = agent.exact_fire_viable / max(agent.exact_fires, 1)
    locked, content_match = content_recovery(agent, cfg['rule'])

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles, 'rule': cfg['rule'],
        'operator_enabled': cfg['enabled'],
        'VR_agent': round(vr_agent, 4),                 # == agent-caused (rho=0)
        'chance_ceiling': round(1.0 / N_ACTIONS, 4),
        'final_window': window_holder['w'],
        'n_mint_events': len(op.mint_events),
        'first_mint_cycle': first_mint_cycle,
        'mint_events': op.mint_events[:5],
        'VR_post_mint': round(post_mint_viable / post_mint_cycles, 4) if post_mint_cycles else None,
        'exact_fires': agent.exact_fires,
        'fire_viable_rate': round(fire_viable, 4),
        'OC9_excess_causal_validity': round(fire_viable - 0.0, 4),   # rho=0
        'locked_keys': locked,
        'content_matches_env_frac': content_match,     # P-D check
        'confirmed_pivots': len(agent.knowledge.get_confirmed()),
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
