# Pre-Registration — R5: is R4's harm REACHABLE through PAO's own channels, or only when planted?

> **FROZEN 2026-06-15, before the N=8 run.** Held-out seeds 200–207 (R4 used 0/100). Code:
> `exp_pao_coverage/r5.py` (reuses `r4.py`/`grid.py`). A deliberate self-critique of R4: R4's positive
> result used *planted* confidently-wrong junk; R5 asks whether that harm arises through PAO's real
> channels without planting.

## 1. Claim under test
R4 found competence-gating helps an ADDITIVE skill-combiner when junk is *confidently-wrong* (mis-tagged
competent skills), and that *weak/immature* junk did not corrupt the sum. R5 decomposes "junk" into the
two channels PAO actually has and asks which, if any, makes the harm reachable:
- **(A) crystallization volume** — over-crystallizing many *immature, correctly-tagged* snapshots;
- **(B) mis-triggering** — a *competent* skill firing in the *wrong* niche (rate ε; count m simultaneous).

## 2. Design (same grid env + base/skills as R4)
- **Condition A:** junk = immature self-snapshots taken during a specialist's own training (own-niche
  tag), `cap ∈ {0,1,2,4,8}` per niche. Arms noskill/ungated/gated (gate = competence ≥0.9).
- **Condition B-rate:** all skills competent & correctly tagged; with prob ε per episode, ONE wrong-niche
  competent skill co-fires. Sweep ε ∈ {0,0.1,0.25,0.5,1.0}.
- **Condition B-count:** with prob 1.0, m wrong-niche competent skills co-fire (with replacement). Sweep
  m ∈ {0,1,2,4,8}. (This is the dimension R4 actually varied via `cap`.)
Metric: greedy mean success over K niches, 30 ep/niche.

## 3. Sanity (1 seed) & integrity
A: ungated = gated = 1.000 at every cap (immature own-niche junk never corrupts the sum). B-rate:
1.00→0.98→0.96→0.88 over ε 0→1 (single mis-fire mostly outvoted). B-count: 1.00→0.88→0.61→0.19→0.09
over m 0→8 (matches R4's cap-8 collapse to ~0.03). Design/metrics frozen here; N=8 confirms.

## 4. Pre-registered predictions (on record)
- **A — VOLUME IS HARMLESS:** ungated ≈ gated, no growth with cap (Spearman(cap, Δ) ≈ 0, gating not
  significant). PAO's crystallization channel produces harmless junk; admission-gating is moot there.
- **B-rate — SINGLE MIS-FIRE IS MILD:** success declines with ε but stays high (>0.8 at ε=1; Spearman<0).
- **B-count — HARM SCALES WITH SIMULTANEOUS WRONG COUNT:** steep monotonic collapse
  (Spearman(m, succ) ≈ −1), success ≪ floor by m=8.

## 5. Interpretation locked in advance
If the predictions hold, the synthesis is: **R4's catastrophe is reachable only when MANY wrong skills
fire at once** (additive sum of confident-wrong biases) — not from crystallization volume per se, nor
from occasional single mis-fires. So PAO's real liability is **indiscriminate fire-all-admitted
triggering of a junk-laden library**, and competence-gated admission helps precisely (and only) by
bounding the fire-set. This *deflates* R4's apparent strength (the harm needed a pessimal many-wrong-fire
setup) while sharpening the actionable lesson: the lever worth engineering is the **router/applicability**
(how many and which skills fire), not the crystallization counter. I revise if A shows a growing gating
gap, or B-count fails to collapse.
