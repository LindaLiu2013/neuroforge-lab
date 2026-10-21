"""v1.6 M10 verification: internalization threshold under TIME-VARYING rho.

Pre-registered in docs/tufe-design-language-v03.md (section v1.6) BEFORE any run.

v1.5 (static rho) showed high environmental supply kills agent-caused
capability, but could not show the TRANSITION: does an agent that grew up at
the vent (high rho) begin forming real capability the moment rho drops? This is
the LUCA "leaving the vent" moment and the one untested LUCA-distilled model
(M10: internalize function r iff (1-rho_r)*l_r > c_r) that makes a sharp, novel
prediction.

Two competing hypotheses this run must separate:
  H-M10 (recoverable internalization): after the rho drop, phase-2 agent-caused
        VR RISES and exceeds the static-high same-window baseline.
  H-critical-period (irreversible): a high-rho childhood permanently prevents
        capability; after the drop agent-caused VR stays ~= static-high (~0).

Arms (rho schedule high -> low at drop_cycle = n//2):
  static_high     0.8 -> 0.8   vent-child that never leaves (phase-2 baseline)
  static_low      0.0 -> 0.0   never had a vent (phase-2 upper reference)
  drop_h2l        0.8 -> 0.0   leaves the vent
  drop_h2l_soft   0.8 -> 0.2   partially retains the vent (LUCA-gradual)

OC-9 discipline: learning is judged ONLY by VR_agent_caused and the excess
causal validity (phase-2 fire-viable - background rho_low). VR_performance and
pivot COUNTS are reported but are NOT the falsification metric.

Usage: python poc/v16_m10_internalization.py <arm> <seed> <n_cycles>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexLoopDomain

# arm -> (rho_high, rho_low)
ARMS = {
    'static_high': (0.8, 0.8),
    'static_low': (0.0, 0.0),
    'drop_h2l': (0.8, 0.0),
    'drop_h2l_soft': (0.8, 0.2),
}


def skel_key_b4(raw_obs):
    """M6 RG fixed-point skeleton: 4-bin (match_rate, reject_rate)."""
    mr = raw_obs.get('match_result', {})
    if not isinstance(mr, dict):
        return 'na'
    m = mr.get('match_rate', 0.0)
    r = mr.get('reject_rate', 0.0)
    return f"m{min(int(m*4),3)}_r{min(int(r*4),3)}"


class RegexVentDomain(RegexLoopDomain):
    """LUCA-faithful food set: with prob p_supply per cycle the ENVIRONMENT
    grants viability independent of the agent's action. Records agent-caused
    viability separately from the final (environment-overridden) viability.
    p_supply is mutable so the runner can flip the schedule at drop_cycle."""

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
        result = type(result)(
            success=final_viable,
            post_state={**result.post_state, 'viable': final_viable},
            metadata={**result.metadata, 'agent_viable': agent_viable,
                      'vent_rescue': vent_rescue},
        )
        return result

    def name(self) -> str:
        return "RegexVent"


def snapshot(agent, domain, cycle):
    """Capture cumulative counters so phase deltas can be computed."""
    ks = agent.knowledge.stats()
    return {
        'cycle': cycle,
        'viable_cycles': agent.viable_cycles,
        'agent_viable_cycles': domain.agent_viable_cycles,
        'total_cycles_seen': domain.total_cycles_seen,
        'exact_fires': agent.exact_fires,
        'exact_fire_viable': agent.exact_fire_viable,
        'confirmed_pivots': ks['n_confirmed'],
        'feasible_volume': agent.feasible_space.stats()['current_volume'],
    }


def rate(dnum, dden):
    return round(dnum / dden, 4) if dden else 0.0


def main():
    arm = sys.argv[1]
    seed, n_cycles = int(sys.argv[2]), int(sys.argv[3])
    rho_high, rho_low = ARMS[arm]
    drop_cycle = n_cycles // 2
    mid_post = drop_cycle + (n_cycles - drop_cycle) // 2

    domain = RegexVentDomain(seed=seed, task_duration=500,
                             dissipation_rate=0.0, p_supply=rho_high)
    agent = TUFEAgent(domain, seed=seed, exact_addressing=True,
                      address_key_fn=skel_key_b4)

    t0 = time.time()
    domain.initial_perturbation()
    snaps = {}
    for c in range(n_cycles):
        if c == drop_cycle:
            domain.p_supply = rho_low
            snaps['drop'] = snapshot(agent, domain, c)
        elif c == mid_post:
            snaps['mid_post'] = snapshot(agent, domain, c)
        agent.run_cycle()
    snaps['end'] = snapshot(agent, domain, n_cycles)
    seconds = time.time() - t0

    d, m, e = snaps['drop'], snaps['mid_post'], snaps['end']

    # phase-2 (post-drop) whole-window agent-caused VR
    p2_seen = e['total_cycles_seen'] - d['total_cycles_seen']
    p2_agent_vr = rate(e['agent_viable_cycles'] - d['agent_viable_cycles'], p2_seen)
    # phase-2 fire-viable and OC-9 excess causal validity
    p2_fires = e['exact_fires'] - d['exact_fires']
    p2_fire_viable = rate(e['exact_fire_viable'] - d['exact_fire_viable'], p2_fires)
    p2_excess = round(p2_fire_viable - rho_low, 4)
    # within-phase-2 rise: late half vs early half agent-caused VR
    e_seen = m['total_cycles_seen'] - d['total_cycles_seen']
    l_seen = e['total_cycles_seen'] - m['total_cycles_seen']
    p2_early_agent_vr = rate(m['agent_viable_cycles'] - d['agent_viable_cycles'], e_seen)
    p2_late_agent_vr = rate(e['agent_viable_cycles'] - m['agent_viable_cycles'], l_seen)
    # phase-1 (pre-drop) agent-caused VR for reference
    p1_agent_vr = rate(d['agent_viable_cycles'], d['total_cycles_seen'])

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles,
        'rho_high': rho_high, 'rho_low': rho_low, 'drop_cycle': drop_cycle,
        'p1_agent_vr': p1_agent_vr,
        'p2_agent_vr': p2_agent_vr,
        'p2_early_agent_vr': p2_early_agent_vr,
        'p2_late_agent_vr': p2_late_agent_vr,
        'p2_agent_vr_rise': round(p2_late_agent_vr - p2_early_agent_vr, 4),
        'p2_performance_vr': rate(e['viable_cycles'] - d['viable_cycles'], p2_seen),
        'p2_fire_viable': p2_fire_viable,
        'p2_excess_causal_validity': p2_excess,
        'p2_fires': p2_fires,
        'end_confirmed_pivots': e['confirmed_pivots'],
        'end_feasible_volume': e['feasible_volume'],
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
