# Draft: correction addendum for the public v1.0 Release page

> **Status: DRAFT — not published.** Editing the public GitHub Release page is a
> public-facing action and needs Jayson's explicit go-ahead + working network.
> This file is the exact text to paste as a Release edit (or as a follow-up
> `v1.0.1` note). Kept locally so the correction is ready and auditable.

---

## Correction & honest re-framing (added after external peer review, Oct 2026)

The original release note reported viability rates for four domains **without
baselines**. That was misleading, and we are correcting it rather than editing
the numbers away.

**Headline correction.** TUF-E v1.0 is **a research framework with an open
hypothesis — not a validated learner.** The substantive contribution is the
theory (pivot three-layer ontology, dissipative knowledge maintenance,
cross-observer consensus) and the `DomainSpec` interface. The bundled demo
domains do **not** yet show that pivot-based learning beats simple controls.

**What a peer review found (5000 cycles, seed 0), and we verified against source:**

| Domain | TUF-E | Random | No-op | Best fixed action |
|---|---|---|---|---|
| Arithmetic | 0.305 | 0.136 | 0.020 | 0.339 (always-add) |
| Ising | 0.451 | 0.657 | 0.413 | 0.349 |
| NK | 0.966 | 0.981 | 0.986 | 0.969 |

- **Arithmetic**: the target answer is never shown to the agent, so viability is
  chance-capped near 1/3; a fixed "always-add" action matches/beats the agent.
- **Ising**: the agent's viability is **below** a random-flip baseline; the
  consensus mechanism currently *hurts* in this continuous domain.
- **NK**: the viability band `[0.3, 0.7]` is passed by random and do-nothing
  policies (~98%), so the 0.966 rate proves nothing about learning.

**Engine honesty.** The released default path is effectively a **state-blind
bandit**: a preference vector indexed only by `action_id`, updated by a binary
viability multiplier. State-conditional pivot machinery exists
(`exact_addressing`) but is off by default, and the pivot-guided branch fires
only ~40% of the time. "Viability, not reward" was therefore overstated for the
shipped configuration.

**"Phase transitions".** The large transition counts in the original note
(e.g. Arithmetic 4319, Ising 32805) are **Betti-number churn** — any change in
`b0`/`b1` was logged — not validated topological reorganization. They should not
be read as evidence of insight events.

**Engineering fixes shipped with this correction:**
- Report key is `report['viability_rate']` (a flat float). The original note's
  `report['viability']['rate']` raises `KeyError`.
- `pyproject.toml` `Development Status` corrected from `4 - Beta` to
  `2 - Pre-Alpha`.
- README now lists the **Regex** domain (previously omitted) and states every
  viability rate beside its baselines.

**What we are NOT doing:** we are not tuning the agent to beat these baselines.
That would be exactly the teleology/reward-hacking drift we have been removing.
The fix is to make the *tasks* learnable (show the target, tighten bands) and the
*metrics* meaningful, then report the outcome honestly — **including if a
state-dependent pivot learner still loses.**

**Next:** `v1.1` is pre-registered (committed before results) to demonstrate one
domain where a state-dependent pivot learner beats random, no-op, best-fixed-action,
**and** a state-blind bandit, all five reported together.

We thank the reviewer. The critique converges with our own post-v1.0 findings and
raised the credibility bar for v1.1.
