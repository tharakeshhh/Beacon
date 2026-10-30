# Beacon: Predictive Student Early-Warning & Retention System

# [Beacon: Predictive Student Early-Warning & Retention System](https://claude.ai/public/artifacts/a9602651-bb40-4a25-933f-560dca0cbab6)
Beacon flags at-risk students from *behavioural movement* rather than lagging
grade/attendance reports, so institutions get a warning while there is still
time to act.

## What it does

| Layer | Module | Idea |
|---|---|---|
| Signals | `beacon/signals.py` | 9 behavioural signals: attendance, coursework, marks, platform activity, peer connection, fee arrears, wellbeing, late submissions, campus access |
| Scoring | `beacon/scoring.py` | Each signal is scored on **level and velocity** (velocity weighted higher, so a fast decline is caught before the level looks bad). Signals are combined as independent evidence with **noisy-OR**, so one severe signal isn't buried by healthy ones |
| Explainability | `scoring.explain` | Additive attribution in hazard space: every score splits into per-signal shares that sum to 1, rendered as plain-language lines a mentor can act on |
| Routing | `beacon/routing.py` | Recommendations go to the **root cause** (e.g. fee-instalment plan, not another attendance contract, when financial strain drives the pattern). Outcomes feed back into Beta posteriors that re-rank interventions |
| Triage | `beacon/triage.py` | Ranks by **expected students retained per mentor-hour**: `risk x effect x responsiveness_decay / hours`, not by raw risk |
| Fairness | `beacon/fairness.py` | The model takes **zero demographic inputs**. A separate audit panel (group labels used only after the fact) compares flag rate, precision and false-positive rate to separate a true institutional finding from a modelling flaw |

### Why noisy-OR

```
P(at risk) = 1 - (1 - leak) * prod_i (1 - p_i)
hazard h_i = -ln(1 - p_i)        # additive:  -ln(1 - P) = sum_i h_i
```

An average of 9 signals dilutes one collapsing signal. Noisy-OR doesn't.

### Fairness verdicts

- Higher flag rate **and** comparable precision -> **institutional finding**: the need is real.
- Higher flag rate **and** lower precision **and** higher false-positive rate on healthy students -> **modelling flaw**: fix the model.

(Precision alone moves with base rate, which is why the false-positive rate is checked too.)

## Quick start

```bash
git clone https://github.com/<you>/beacon-early-warning.git
cd beacon-early-warning
pip install -r requirements.txt
pytest -q
python demo.py
```

`demo.py` generates a synthetic cohort, then prints detection metrics, a
plain-language explanation for one flagged student, a triage comparison under a
fixed mentor-hour budget, the feedback loop re-ranking interventions, and the
fairness audit.

```python
import numpy as np
from beacon import score_student, explain, Router

X = np.load("student_weekly_signals.npy")   # shape (weeks, 9), 1 = healthy
result = score_student(X)
print(result.score, result.top_cause)
print("\n".join(explain(result)))
print(Router().recommend(result.top_cause))
```

## A note on numbers

This repo runs on a **synthetic cohort** (`beacon/synthetic.py`), so the demo's
metrics are illustrative and will not match the hackathon figures (median 35
days of warning; 65 of 74 hidden at-risk students caught), which came from the
hackathon dataset. Swap in your own data, keep thresholds in `ScoreConfig`, and
report numbers from your own evaluation.

## Project layout

```
beacon/        signals, scoring, routing, triage, fairness, synthetic data
tests/         unit tests for each layer
demo.py        end-to-end run
```

## License

MIT
