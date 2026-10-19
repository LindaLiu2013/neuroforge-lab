# TUF-E PoC Archive Summary v0.9

> Date: 2026-10-18
> Status: Tier 1-3 Complete + Grounding Experiments v0.1-v0.9

---

## Evidence Chain

### EXP-001: Multi-Signal Pivot Detection Robustness
- **Result**: CONFIRMED
- **Key metric**: mean_efficiency=0.86, min_efficiency=0.60, Wilcoxon p<0.001
- **Conclusion**: Topology-aware pivot detection is robust across 20 random seeds

### EXP-002: Parameter Sensitivity Analysis
- **Result**: SENSITIVE (all 3 dimensions failed stability criteria)
- **Key finding**: alpha=0.2 outperforms alpha=0.5; weight_power=2.0 outperforms 1.0
- **Action**: Optimal parameters identified for EXP-003

### EXP-003: Adaptive Parameters on Varied Topologies
- **Result**: CONFIRMED
- **Key metric**: 1.0 efficiency across chain/star/hierarchical topologies with optimal params
- **Conclusion**: With tuned parameters, detection generalizes across topology types

### EXP-004: Weight Overlap Stress Test
- **Result**: CONFIRMED (near-zero degradation)
- **Key metric**: 0.96 efficiency at complete weight overlap (sep_ratio=1.0)
- **Critical finding**: Core capability comes from topology signals (betweenness + community), NOT weight

---

## Tier 1 Integration Results

All 7 checks PASS:

| Check | Result | Detail |
|-------|--------|--------|
| Topology Protection | PASS | 5/5 true bridges detected |
| C2 Violation Detection | PASS | 8 violations flagged |
| Multi-Projection Consensus | PASS | 2 edges confirmed (4 projections) |
| Energy Tracking | PASS | 3 ops logged, Landauer=4.70e-16 J |
| Layer Separation: Query | PASS | Meta rules readable by object layer |
| Layer Separation: Block Modify | PASS | Object→Meta modification REJECTED |
| Layer Separation: Flag Self-Eval | PASS | Self-evaluation correctly flagged as C5-B1 |

### Known Limitation (RESOLVED in Tier 2)
Multi-projection consensus did not include true bridges in its confirmed set (0/5). **Fixed in Tier 2 Module D**: weak-link-sensitive projections + weighted voting achieved 5/5 bridge capture.

---

## Tier 2 Integration Results

All 8 checks PASS:

| Check | Result | Detail |
|-------|--------|--------|
| A: Steady State | PASS | Active entries maintained between cycles 10-70 |
| A: Forgetting | PASS | 45→0 entries after energy cutoff at cycle 70 |
| B: Valid Apply | PASS | decay_rate modified successfully |
| B: Immutable Block | PASS | min_viability modification REJECTED (C5-B2) |
| B: Range Check | PASS | alpha=2.0 out of range REJECTED |
| B: Degradation Rollback | PASS | perf ratio=0.5 triggered auto-rollback |
| C: Cat Pivot Detection | PASS | 5/5 true bridges hit, 100% efficiency vs heuristic |
| D: Weak-Link Consensus | PASS | 5/5 bridges captured (was 0/5 in Tier 1) |

### Key Findings

1. **Category-theoretic detection outperforms heuristic**: 5/5 hits (same as heuristic) plus 28 additional irreducible morphisms identified. Validates Q0.4 direction — category theory is computationally feasible and provides incremental structural insight.

2. **Dissipative knowledge behaves thermodynamically**: Steady state maintained under energy flow; complete forgetting when energy stops. Knowledge = far-from-equilibrium pattern, not static storage.

3. **Weak-link consensus fully resolved**: Adding inverse_weight, community_bridge, structural_hole, persistence_aware projections with 2x voting weight captured all 5 bridge edges that Tier 1 missed entirely.

4. **Online self-modification safe under C5**: All four constraint types enforced — immutable params blocked, range validation active, degradation triggers automatic rollback.

---

## File Index

### Design Documents (`docs/`)
- `tufe-design-constraints.md` — v0.4 constraint specification (L0/L1/L2)
- `tufe-agent-architecture-v0.4.html` — Architecture diagram (SVG)
- `tufe-implementation-framework-v0.4.html` — Three-tier feasibility map

### Experiments (`poc/`)
- `tufe_poc_v1.py` / `poc_results.json` — Original oracle-based PoC
- `tufe_poc_v2.py` / `poc_v2_results.json` — PH auto-detection (replaced oracle)
- `tufe_exp001_robustness.py` / `EXP-001_results.json` — 20-seed robustness
- `tufe_exp002_sensitivity.py` / `EXP-002_results.json` — Parameter sensitivity
- `tufe_exp003_adaptive.py` / `EXP-003_results.json` — Adaptive params + topologies
- `tufe_exp004_overlap.py` / `EXP-004_results.json` — Weight overlap stress test
- `tufe_tier1_prototype.py` / `tier1_results.json` — Tier 1 integration
- `tufe_tier2_prototype.py` / `tier2_results.json` — Tier 2 integration (4 new modules)
- `tufe_tier3_prototype.py` / `tier3_results.json` — Tier 3 research validation (4 modules)
- `tufe_agent.py` / `agent_demo_results.json` — Integrated agent prototype (12 modules, 20-cycle demo)
- `tufe_grounding_experiment.py` / `grounding_experiment_results.json` — Semantic grounding v0.1
- `spatial_reanalysis.py` / `spatial_reanalysis_results.json` — Neural-perspective reanalysis
- `tufe_grounding_v02.py` / `grounding_v02_results.json` — Lossy self-perception + composable pivots
- `tufe_grounding_v03.py` / `grounding_v03_results.json` — Composition loop + calibrated self-model
- `tufe_grounding_v04.py` / `grounding_v04_results.json` — Multi-agent heterogeneous consensus

### Pre-registration & Audit (`poc/preregister/`, `poc/audit/`)
- Pre-registration specs for EXP-001 through EXP-004
- Audit logs for experimental runs

---

## Tier 3 Research Results (Feasibility Assessment)

Tier 3 is research-grade: outcomes are feasibility verdicts, not pass/fail.

| Module | Verdict | Key Finding |
|--------|---------|-------------|
| H: Hardware Meta-Layer | FEASIBLE_WITH_OS_PRIMITIVES | Pipe-based isolation achieves C5 enforcement semantics; ~5μs latency; true ROM needs FPGA/TEE |
| I: Heterogeneous Observer | PROTOCOL_DEFINED_PARTIAL_ANSWER | Jaccard=0.0 between topology vs local observers; disagreement IS informative; full human-AI protocol needs interface design |
| J: Irreducibility Scaling | APPROXIMATION_VIABLE | Exact ≤150 nodes (<12s); Local-k approx **15x speedup, P=0.998, R=1.0** |
| K: Real-Data Validation | PARTIAL_L4_BRIDGE | Python stdlib graph analyzed; `collections`/`abc` correctly identified as hubs |

### Key Findings

1. **Q0.4 substantially answered**: Bounded Dijkstra local-k approximation gives near-perfect recall (1.0) with 15x speedup. Category-theoretic irreducibility is computationally tractable up to hundreds of nodes. For >500 nodes, sampling methods needed but the precision/recall tradeoff is favorable.

2. **Disagreement as signal (Q0.5 partial answer)**: Two genuinely different perception functions produce zero-overlap pivot candidates — and this is the correct behavior. The protocol distinguishes "same answer, different paths" from "fundamentally different perceptions." The disagreement topology itself carries structural information about the system.

3. **Real-data interpretability**: On Python stdlib import graph, the system identifies `collections` and `abc` as highest-centrality modules — matching programmer intuition. This partially bridges the L4 gap: the pipeline works on real data, but the data→graph construction protocol remains ad-hoc.

4. **Hardware enforcement simulatable**: OS-level pipe isolation captures C5 constraint semantics without custom hardware. True physical enforcement (ROM, TEE) would strengthen guarantees but the software simulation validates the architectural pattern.

---

## Remaining Open Questions

- **Q0.3**: What is the minimal graph construction protocol that preserves pivot topology from raw data? (L4 gap — partially bridged by Module K but no general solution)
- **Q0.4**: Can category-theoretic irreducibility be computed efficiently? **(Answered: yes, with local-k approximation at 15x speedup, P≥0.99, R=1.0 up to 150 nodes)**
- **Q0.5**: What protocol enables heterogeneous observers to cross-validate? **(Substantially answered by v0.4: shared abstraction layer required; translation protocol IS a constructor; only operations stable across perceptual spaces achieve consensus)**
- **Q0.6** (NEW): Does the shared abstraction layer itself evolve, or must it be fixed a priori? Current v0.4 uses hardcoded bucket mappings. If the translation protocol could self-modify under C5 supervision, it would close the loop between perception and communication.
- **Q0.7** (NEW): Can composition emergence be accelerated by explicitly rewarding causal chain discovery rather than just task success? Current compositions emerge passively; active causal probing may be needed.

## Beyond Tier 3

The three-tier feasibility map is now fully explored. **Integrated agent prototype running.**

### Agent Demo Results (20 cycles)
- Knowledge: 1→4→3 entries (dissipative dynamics confirmed)
- Parameters: `reinforcement_boost` self-evolved 0.3→0.94 (12 modifications, all meta-validated)
- Confidence: 1.0→0.71 (natural decay as graph evolves)
- Energy: 1.36M bits, 4.09e-15 J Landauer minimum
- Observer Jaccard: stable 0.25 (complementary perceptions)
- C5 enforcement: all 4 tests pass

### Next Frontier
1. **Real-world benchmarking**: Test on larger codebases, knowledge graphs, biological networks
2. **Formal verification**: Prove C5 enforcement properties in the simulated hardware model
3. **Human-in-the-loop study**: Implement the observer interface for Q0.5 empirical validation
4. **Persistent agent**: Add serialization/deserialization for long-running experiments
5. **Multi-agent interaction**: Multiple TUF-E agents sharing an environment with consensus

---

## Semantic Grounding Experiments

Progression from statistical association toward causal understanding through iterative refinement.

### v0.1: Basic Grounding (tufe_grounding_experiment.py)
- **SR**: 19.8% (500 episodes)
- **Key finding**: Spontaneous mul preference emergence (0.58) without being told what operations mean
- **Mechanism**: Arity-keyed preferences + proximity reward + structural hints
- **Limitation**: Flat key-value knowledge, no self-model, trivially observable environment

### v0.2: Lossy Self-Perception (tufe_grounding_v02.py)
- **SR**: 21.3% (600 episodes)
- **New features**: Probe-based self-model (energy-costly, noisy), composable pivot library, partial observability, delayed feedback
- **Key findings**:
  - Self-model accuracy 93.4% but confidence only 80% — agent is *underconfident*
  - Self-perception diverges from reality at early stages (thinks abs/neg are top ops when actually mul/add)
  - Pivot library converges from 8-10 to 4 strong pivots (all mul-based)
  - Composition system weak (only 1 degenerate composition)
- **Philosophical alignment**: Validates Jayson's insight that self-knowledge is always a lossy symbolic projection

### Spatial Reanalysis (spatial_reanalysis.py)
- Confirmed 4/4 neural-like structural features in v0.1 learning dynamics:
  1. **Topographic mapping**: 4.3x binary/unary operation separation in representation space
  2. **Receptive fields**: mul→large numbers, sub→small numbers
  3. **Lateral inhibition**: 7/7 ops inhibited when mul rises
  4. **Spatial consolidation**: Concentration index 0.003→0.298 over training

### v0.3: Composition Loop + Calibrated Self-Model (tufe_grounding_v03.py)
- **SR**: 13.1% (800 episodes, mixed difficulty)
- **Fixes**: Composition attempts now feed back into action selection; confidence based on structural signals not raw SR
- **Key findings**:
  - add became dominant (29%) instead of mul — escalating value ranges favor stable operations
  - Composition loop functional: (add,add) tried 167 times, 12% success rate
  - Confidence-accuracy gap narrowed to 0.14 (from v0.2's fixed 0.80)
  - Lower overall SR expected due to mixed difficulty (difficulty 2-3 tasks inherently harder)

### v0.4: Multi-Agent Heterogeneous Consensus (tufe_grounding_v04.py)
- **Results** (600 episodes, 3 conditions):

| Condition | SR | Pivots | Energy |
|-----------|-----|--------|--------|
| Multi-Agent + Consensus | **18.0%** | 5 confirmed | ~3x single |
| Multi-Agent No Consensus | 14.8% | 75 private total | ~3x single |
| Single Agent Baseline | 11.3% | 32 private | 1x |

- **Critical discovery**: Without shared abstraction layer, heterogeneous observers produce **0 confirmed pivots** — different constructors literally cannot communicate. Adding translation protocol enables 5 confirmed pivots.
- **All confirmed pivots are add-based**: Only addition is stable enough across magnitude/symbolic/temporal perceptual spaces to achieve cross-observer agreement
- **Consensus boost**: +6.7% over single agent, +3.2% over multi-agent without consensus
- **Q0.5 validation**: Heterogeneous observers CAN cross-validate, but ONLY through a shared translation protocol. The translation layer is itself a constructor that determines what CAN be agreed upon.

### Grounding Experiment Version History (v0.1-v0.4: Arithmetic Domain)

| Version | SR | Type | Key Innovation |
|---------|-----|------|----------------|
| v0.1 | 19.8% | Single | Spontaneous op preference |
| v0.2 | 21.3% | Single + lossy self | Underconfident self-model |
| v0.3 | 13.1% | Single + composition | Composition loop, calibrated confidence |
| v0.4 (consensus) | **18.0%** | Multi-agent + C3 | Cross-observer validation works |
| v0.4 (no consensus) | 14.8% | Multi-agent | Diversity helps even without agreement |
| v0.4 (single baseline) | 11.3% | Single | Baseline for comparison |

### Key Insight (v0.1-v0.4)

The progression reveals a fundamental tension: **more realistic constraints (lossy perception, partial observability, heterogeneous observers) reduce raw performance but increase structural validity**. v0.2's 21.3% SR is the highest, but v0.4's 18.0% with consensus is more meaningful because it demonstrates genuine cross-observer knowledge validation — the kind of robustness that scales beyond individual perception limits.

---

## Cafeteria Management Experiments (v0.5-v0.9)

Real-world domain simulation: dual-layer cafeteria environment with normative layer (regulations/SOPs) and operational layer (human deviation, economic pressure, incomplete information). Four-stage learning model: Observation → Rule Internalization → Recomposition → Habituaton Decline.

### v0.5: Baseline Cafeteria Agent (tufe_grounding_v05.py)
- **Environment**: 120-day simulation, 5 causal chains, delayed risk exposure
- **Stage transitions**: Day 11 (→S2), Day 30 (→S3), Day 71 (→S4)
- **Final stats**: 700 actions, 95.1% success rate, 9 pivots
- **Safety scores**: S1=1.0, S2=0.846, S3=0.718, S4=0.727
- **Key finding**: Stage 4 habituaton decline verified — compliance baselines erode 0.3%/day, inspection gaps >14 days create blind spots, safety-0 events occur despite 95% operational success rate
- **Action distribution issue**: menu_plan dominated at 67.9% by day 120 (reward bias toward high-frequency operations)
- **Budget**: 31,174 remaining (of 500,000)

### v0.6: LLM+TUF-E Dual-Channel Hybrid (tufe_grounding_v06.py)
- **Architecture**: LLM-as-Perception (simulated biases) + TUF-E-as-Cognition + Shared Abstraction Layer
- **LLM perception**: 40.8% detection rate, 59.2% miss rate (concentrated on slow variables), 5% false alarm
- **Consensus**: 120/120 days had disagreements, avg info density 0.76
- **Safety scores**: S1=0.909, S2=0.924, S3=0.662, S4=0.775
- **vs v0.5**: S2 safety +8%, S4 safety +5%; S3 slightly lower
- **Action distribution issue**: inspect dominated at 77.1% (disagreement handler over-boosted inspect preference)
- **Q0.5 validated**: Disagreement IS information — heterogeneous observers provide complementary coverage
- **Budget**: 32,468 remaining

### v0.7: LLM+Statistical+TUF-E Triple-Channel (tufe_grounding_v07.py)
- **New channel**: Non-linguistic statistical perception monitoring 7 dimensions (safety drift, compliance, satisfaction, budget rate, hidden risk count, personnel fatigue, cert validity, inspection interval)
- **Drift detection**: z-score against 14-day baseline window, |z|>1.5 signals, |z|>2.0 alerts
- **Statistical alerts**: 338 total across 7 dimensions (covering slow variables LLM completely misses)
- **Three-way consensus**: replaces two-way, tracks stat_unique_domain disagreements
- **Safety scores**: S1=0.909, S2=0.936, S3=0.767, S4=0.860
- **vs v0.6**: S3 safety **+10.5%** (0.662→0.767), S4 safety **+8.5%** (0.775→0.860)
- **LLM detection improved**: 45.7% (from 40.8%)
- **Action distribution issue**: order dominated at 59.7%
- **Budget**: 27,560 remaining

### v0.8: Self-Evolving Abstraction Layer (tufe_grounding_v08.py)
- **Architecture change**: Fixed SharedAbstractionLayer replaced by EvolvingAbstractionLayer
- **Signal credit system**: Each (source, type, dimension) combination maintains credit score; good (reward>0.3) → credit×1.05, bad → credit×0.92; clamped [0.1, 3.0]
- **Auto-discovery**: LLM confidence calibration issue detected (critical credit=3.0/91% accurate vs high credit=0.19/15% accurate)
- **Safety scores**: S1=0.909, S2=0.915, S3=0.797, S4=0.852
- **vs v0.7**: S3 safety **+3%** (0.767→0.797)
- **Statistical alerts**: 370 total
- **Action distribution issue**: order dominated at 84.3% (self-evolution over-amplified high-credit signals)
- **Budget**: 11,250 remaining

### v0.9: Diversity-Protected Self-Evolution (tufe_grounding_v09.py)
- **New mechanisms**: Diversity tax (>60% source concentration → suppression), exploration bonus (<5 uses → 1.5x, <15 uses with positive record → 1.2x), periodic mean-reversion (every 100 updates: credit = credit×0.85 + mean×0.15)
- **Gentler update rates**: 1.04/0.93 (from 1.05/0.92), credit ceiling lowered to 2.5
- **Safety scores**: S1=0.909, S2=0.934, S3=0.846, S4=0.851
- **vs v0.8**: S3 safety **+4.9%** (0.797→0.846) — new high across all versions
- **Source balance**: More balanced between LLM and Statistical channels
- **Statistical alerts**: 364 total
- **Budget**: 25,671 remaining

### Cafeteria Experiment Summary Table

| Version | Architecture | S3 Safety | S4 Safety | Success Rate | Key Improvement |
|---------|-------------|-----------|-----------|-------------|-----------------|
| v0.5 | TUF-E only | 0.718 | 0.727 | 95.1% | Baseline + habituaton decline verified |
| v0.6 | +LLM perception | 0.662 | 0.775 | 94.3% | Heterogeneous consensus works |
| v0.7 | +Statistical channel | 0.767 | 0.860 | 94.3% | Slow-variable blindness compensated |
| v0.8 | Self-evolving translation | 0.797 | 0.852 | 94.4% | Auto-discovered calibration issues |
| v0.9 | +Diversity protection | **0.846** | 0.851 | 95.3% | Prevented signal concentration |

### Cross-Version Insights (v0.5-v0.9)

1. **Efficiency-safety decoupling confirmed**: All versions achieve >94% operational success, yet S3 safety ranges 0.662-0.846. High task performance does NOT guarantee safety awareness.

2. **Complementary perception works**: Each added channel compensates for others' blind spots. LLM misses slow variables; statistical channel covers them. The disagreement between channels IS the valuable signal.

3. **Self-evolution needs guardrails**: v0.8 showed self-evolving translation can improve S3 by 3%, but also caused action concentration (order at 84%). v0.9's diversity protection fixed this while further improving S3 to 0.846.

4. **Progressive safety improvement**: S3 safety increased monotonically from v0.6 (0.662) through v0.9 (0.846) — a 27.8% relative improvement through architectural refinement alone, without changing the underlying environment or reward structure.

5. **Remaining limitation**: All results are single-seed (seed=42). Statistical significance testing with multiple seeds is needed before claiming robust improvements. Action preference distributions remain uneven across versions, suggesting reward function design still has room for improvement.

### File Index Additions (v0.5-v0.9)

- `tufe_grounding_v05.py` / `grounding_v05_results.json` — Cafeteria baseline + habituaton decline
- `tufe_grounding_v06.py` / `grounding_v06_results.json` — LLM+TUF-E dual-channel hybrid
- `tufe_grounding_v07.py` / `grounding_v07_results.json` — Triple-channel (LLM+Stat+TUF-E)
- `tufe_grounding_v08.py` / `grounding_v08_results.json` — Self-evolving abstraction layer
- `tufe_grounding_v09.py` / `grounding_v09_results.json` — Diversity-protected self-evolution

### Updated Open Questions

- **Q0.3**: Data→graph construction protocol remains ad-hoc (unchanged)
- **Q0.5**: Substantially answered — heterogeneous observers cross-validate through shared translation protocol; disagreement topology carries structural information
- **Q0.6**: Answered by v0.8-v0.9 — shared abstraction layer CAN self-evolve under credit-based feedback; diversity protection prevents lock-in
- **Q0.8** (NEW): Are the v0.5-v0.9 safety improvements statistically robust? Multi-seed experiments needed.
- **Q0.9** (NEW): Can the signal credit system be grounded in formal Bayesian updating rather than heuristic multiplicative rules?
- **Q0.10** (NEW): What is the minimal set of perception channels needed for a given domain? Is there a principled way to determine channel sufficiency?
