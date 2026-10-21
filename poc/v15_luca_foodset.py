"""v1.5 M13 verification: environmental food-set reliability rho vs LEARNING.

Distilled from LUCA through four lenses (Lewin B=f(P,E), entropy/dissipative
structures, Bayes degeneracy, Fermat least-internal-work). M13 predicts the
core TUF-E decoupling: PERFORMANCE (VR) is monotone non-decreasing in the
environmental supply reliability rho, but LEARNING (confirmed pivots /
internalized structure) is INVERTED-U:
  rho->0  discovery impossible, agent never reaches viable => no pivots confirm
  rho->1  viability is environment-caused, agent action no longer predictive
          => pivot triplets stop reproducibly yielding viable => consensus fails
  rho*    environment starts, agent completes => action IS causal => pivots peak

Falsification: if confirmed_pivots / feasible volume rise monotonically with
rho (same shape as VR), M13's learning-performance decoupling is wrong and the
naive M8 "more food set = better" (monotone) view stands.

Food set F modeled as task-AGNOSTIC environmental catalysis (LUCA's vent
metals supplying reactive species, NOT the answer): each cycle, with prob
rho the environment appends one random primitive from a fixed catalyst set
for free (no agent action consumed), replacing dissipation. Generic supply =>
not reward hacking; whether a supplied token helps depends on the task.

psi fixed at the M6 RG fixed point (b=4 skeleton) so maintenance works and we
isolate the food-set effect on discovery/learning.

Usage: python poc/v15_luca_foodset.py <rho_x100> <seed> <n_cycles>
  e.g. rho_x100=20 means rho=0.20
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexLoopDomain

CATALYSTS = [r'\d', '[A-Z]', r'[^\w\s]', r'\s']


def skel_key_b4(raw_obs):
    """M6 RG fixed-point skeleton: 4-bin (match_rate, reject_rate)."""
    mr = raw_obs.get('match_result', {})
    if not isinstance(mr, dict):
        return 'na'
    m = mr.get('match_rate', 0.0)
    r = mr.get('reject_rate', 0.0)
    return f"m{min(int(m*4),3)}_r{min(int(r*4),3)}"


class RegexVentDomain(RegexLoopDomain):
    """LUCA-faithful food set: the vent spontaneously runs reactions.

    With prob rho per cycle the ENVIRONMENT grants viability independent of the
    agent's action (like the vent spontaneously running 95-97% of LUCA's
    reactions). We record BOTH the final viability the agent experiences
    (performance) and the agent-caused viability before the environmental
    override (true learning signal). At high rho the agent experiences viability
    it did not cause => its action->viability attribution becomes noise =>
    confirmed pivots are spurious and collapse when the vent does not rescue.
    """

    def __init__(self, seed=42, task_duration=500, dissipation_rate=0.0, p_supply=0.0):
        super().__init__(seed=seed, task_duration=task_duration,
                         dissipation_rate=dissipation_rate)
        self.p_supply = p_supply
        self._vent_rng = np.random.default_rng(seed + 1000)
        self.agent_viable_cycles = 0
        self.total_cycles_seen = 0

    def act(self, action_id: int):
        result = super().act(action_id)
        agent_viable = bool(result.post_state.get('viable', False))
        self.total_cycles_seen += 1
        if agent_viable:
            self.agent_viable_cycles += 1
        vent_rescue = self.p_supply > 0 and self._vent_rng.random() < self.p_supply
        final_viable = agent_viable or vent_rescue
        if final_viable != agent_viable:
            result = type(result)(
                success=final_viable,
                post_state={**result.post_state, 'viable': final_viable},
                metadata={**result.metadata, 'agent_viable': agent_viable,
                          'vent_rescue': vent_rescue},
            )
        else:
            result.metadata['agent_viable'] = agent_viable
            result.metadata['vent_rescue'] = vent_rescue
        return result

    def name(self) -> str:
        return "RegexVent"


def main():
    rho = int(sys.argv[1]) / 100.0
    seed, n_cycles = int(sys.argv[2]), int(sys.argv[3])
    domain = RegexVentDomain(seed=seed, task_duration=500,
                             dissipation_rate=0.0, p_supply=rho)

    per_task = {}
    orig_act = domain.act

    def tracked_act(action_id):
        result = orig_act(action_id)
        acc = per_task.setdefault(result.metadata['task'], [0, 0])
        acc[0] += 1
        acc[1] += 1 if result.metadata.get('agent_viable', result.success) else 0
        return result

    domain.act = tracked_act

    agent = TUFEAgent(domain, seed=seed, exact_addressing=True, address_key_fn=skel_key_b4)
    t0 = time.time()
    report = agent.run(n_cycles=n_cycles, report_every=n_cycles)
    seconds = time.time() - t0

    ptf = {t: round(v / n, 3) for t, (n, v) in sorted(per_task.items())}
    out = {
        'rho': rho,
        'seed': seed,
        'n_cycles': n_cycles,
        'VR_performance': report['viability_rate'],
        'VR_agent_caused': round(domain.agent_viable_cycles / max(domain.total_cycles_seen, 1), 4),
        'LEARN_confirmed_pivots': report['knowledge']['n_confirmed'],
        'LEARN_feasible_volume': report['feasible_space']['current_volume'],
        'replay_fires': agent.exact_fires,
        'replay_fire_viable_rate': round(agent.exact_fire_viable / max(agent.exact_fires, 1), 3),
        'max_task_agent_viable_fraction': max(ptf.values()) if ptf else 0.0,
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
