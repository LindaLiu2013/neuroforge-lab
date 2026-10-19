"""
TUF-E Prototype 1.0 System Validation
3 domains × 5 seeds × 5000 cycles
"""
import json
import time
import sys
sys.path.insert(0, '.')

from tufe import TUFEAgent
from tufe.examples import ArithmeticDomain, IsingDomain, NKDomain

SEEDS = [42, 123, 7, 2024, 999]
N_CYCLES = 5000
REPORT_EVERY = 1000

DOMAINS = {
    'Arithmetic': lambda s: ArithmeticDomain(seed=s),
    'Ising-8x8': lambda s: IsingDomain(L=8, seed=s),
    'NK-N12-K3': lambda s: NKDomain(N=12, K=3, seed=s),
}

results = {}

for domain_name, domain_factory in DOMAINS.items():
    results[domain_name] = []
    print(f'\n{"="*60}')
    print(f'DOMAIN: {domain_name}')
    print(f'{"="*60}')

    for seed in SEEDS:
        t0 = time.time()
        domain = domain_factory(seed)
        agent = TUFEAgent(domain, seed=seed)
        report = agent.run(n_cycles=N_CYCLES, report_every=REPORT_EVERY)
        elapsed = time.time() - t0

        entry = {
            'seed': seed,
            'viability_rate': report['viability_rate'],
            'phase_transitions': report['topology']['n_transitions'],
            'feasible_space_final': report['feasible_space']['current_volume'],
            'feasible_space_peak': report['feasible_space']['peak_volume'],
            'consensability': report['consensus']['consensability'],
            'consensus_strategy': report['consensus']['strategy'],
            'elapsed_sec': round(elapsed, 1),
        }

        # Extract confirmed pivots count from knowledge store
        if hasattr(agent, 'knowledge_store'):
            entry['confirmed_pivots'] = len(agent.knowledge_store.confirmed)
        else:
            entry['confirmed_pivots'] = len(report['knowledge'].get('confirmed', {}))

        results[domain_name].append(entry)
        print(f'  Seed {seed}: VR={entry["viability_rate"]:.3f} | '
              f'Pivots={entry["confirmed_pivots"]} | PT={entry["phase_transitions"]} | '
              f'FS={entry["feasible_space_final"]} | {elapsed:.1f}s')

# Summary statistics
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')

summary = {}
for domain_name, entries in results.items():
    vrs = [e['viability_rate'] for e in entries]
    pts = [e['phase_transitions'] for e in entries]
    pivs = [e['confirmed_pivots'] for e in entries]
    fss = [e['feasible_space_final'] for e in entries if e['feasible_space_final'] is not None]

    import numpy as np
    s = {
        'vr_mean': round(float(np.mean(vrs)), 4),
        'vr_std': round(float(np.std(vrs)), 4),
        'pt_mean': round(float(np.mean(pts)), 1),
        'pivot_mean': round(float(np.mean(pivs)), 1),
        'fs_mean': round(float(np.mean(fss)), 1) if fss else None,
        'seeds': len(entries),
        'cycles_per_seed': N_CYCLES,
    }
    summary[domain_name] = s
    print(f'{domain_name}: VR={s["vr_mean"]:.4f}±{s["vr_std"]:.4f} | '
          f'Pivots={s["pivot_mean"]:.0f} | PT={s["pt_mean"]:.0f} | FS={s["fs_mean"]}')

output = {'experiment': 'TUF-E Prototype 1.0 System Validation',
          'config': {'n_cycles': N_CYCLES, 'seeds': SEEDS},
          'per_seed': results,
          'summary': summary}

with open('validation_results.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print('\nResults saved to validation_results.json')
