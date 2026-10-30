"""End-to-end demo on a synthetic cohort:  python demo.py"""
import numpy as np

from beacon import ScoreConfig, score_student, explain, Router, Case, allocate, audit
from beacon.triage import expected_retained
from beacon.synthetic import generate

CFG = ScoreConfig()
MIN_WEEKS = 6


def first_flag_week(X_i, last_week, cfg=None):
    cfg = cfg or CFG
    for t in range(MIN_WEEKS, last_week + 1):
        if score_student(X_i[:t], cfg).score >= cfg.threshold:
            return t
    return None


def main():
    co = generate()
    n, weeks, _ = co.X.shape

    flag_wk = []
    for i in range(n):
        last = int(co.dropout_week[i]) if co.at_risk[i] else weeks
        flag_wk.append(first_flag_week(co.X[i], last))
    flagged = np.array([w is not None for w in flag_wk])

    tp = flagged & co.at_risk
    print("=== Detection ===")
    print(f"At-risk students:        {co.at_risk.sum()}")
    print(f"Caught before dropout:   {tp.sum()}  (recall {tp.sum() / co.at_risk.sum():.0%})")
    print(f"Precision:               {tp.sum() / max(flagged.sum(), 1):.0%}")
    leads = [(co.dropout_week[i] - flag_wk[i]) * 7 for i in np.where(tp)[0]]
    print(f"Median advance warning:  {np.median(leads):.0f} days")
    hidden = co.at_risk & ~co.failing_mark
    print(f"At-risk with NO failing mark: {hidden.sum()}, caught {(hidden & flagged).sum()}")

    # ---- Explain one case
    i = int(np.where(tp & (co.cause == "financial"))[0][0])
    res = score_student(co.X[i, :flag_wk[i]])
    print(f"\n=== Why student {i} was flagged (score {res.score:.2f}) ===")
    for line in explain(res):
        print(" -", line)

    # ---- Triage at a snapshot week
    SNAP, BUDGET = 18, 30.0
    router = Router(seed=1)
    cases = []
    for i in range(n):
        if co.at_risk[i] and co.dropout_week[i] <= SNAP:
            continue
        r = score_student(co.X[i, :SNAP])
        if r.score < CFG.threshold:
            continue
        rec = router.recommend(r.top_cause)
        first = flag_wk[i] if flag_wk[i] is not None else SNAP
        cases.append(Case(i, r.score, rec.cause, rec.intervention, rec.hours,
                          rec.effect, max(SNAP - first, 0) * 7))
    print(f"\n=== Triage @ week {SNAP}: {len(cases)} flagged, {BUDGET:.0f} mentor-hours ===")
    for strat in ("risk", "value"):
        chosen, used = allocate(cases, BUDGET, strat)
        got = sum(expected_retained(c) for c in chosen)
        print(f"{strat:>5}-ranked: {len(chosen):2d} students, {used:4.1f} h, "
              f"expected retained = {got:.2f}")

    # ---- Feedback loop
    print("\n=== Feedback loop (financial) ===")
    print("before:", [(n_, round(v, 2)) for n_, v in router.ranking("financial")])
    for _ in range(12):
        router.record_outcome("financial", "Bursary referral", True)
    print("after: ", [(n_, round(v, 2)) for n_, v in router.ranking("financial")])

    # ---- Fairness audit (group labels never touched the model)
    print("\n=== Fairness audit ===")
    for row in audit(flagged, co.at_risk, co.group):
        print(f"Group {row['group']}: flag rate {row['flag_rate']:.0%} vs {row['rest_flag_rate']:.0%}, "
              f"precision {row['precision']:.0%} vs {row['rest_precision']:.0%}")
        print("   ->", row["verdict"])


if __name__ == "__main__":
    main()
