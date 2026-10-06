"""Benchmark the 3 patterns on the same cases.
Usage: python evaluate.py [--patterns react hybrid] [--groups A B] [--repeats 3] [--out results.csv]
"""
import argparse
import sys

import pandas as pd

from flight_agent import PATTERNS, Constraints, make_model, run

CORE = ["react", "plan_execute", "hybrid"]      # the 3 required patterns; the rest are bonus (BAOCAO Phụ lục A)
# (id, constraint overrides, expected "DONE" or "FAILED:<handoff reason>", extra run() args). Group = first letter of id.
CASES = [
    ("A1", {}, "DONE", {}),                                                            # A: simple
    ("A2", dict(destination="HAN"), "DONE", {}),
    ("A3", dict(origin="HAN", destination="SGN", date="2026-10-08"), "DONE", {}),
    ("B1", dict(destination="HAN", depart_before="10:00", max_price=1_500_000), "DONE", {}),   # B: many constraints
    ("B2", dict(depart_before="09:00", max_price=1_900_000), "DONE", {}),
    ("B3", dict(origin="HAN", destination="SGN", date="2026-10-08", depart_before="23:59", max_price=1_300_000), "DONE", {}),
    ("C1", dict(max_price=1_000_000), "FAILED:constraints", {}),                       # C: impossible -> handoff
    ("C2", dict(destination="HAN", depart_before="06:00"), "FAILED:constraints", {}),
    ("C3", dict(date="2026-10-09"), "FAILED:constraints", {}),
    ("C4", dict(destination="HAN", max_price=1_200_000), "FAILED:constraints", {}),    # cheap flight is in the afternoon
    ("D1", {}, "FAILED:payment", dict(approve=lambda a: False)),                       # D: human refuses payment
    ("E1", {}, "DONE", dict(fail_search=1)),                                           # E: tool fails once -> recover
    ("E2", {}, "FAILED:tool_error", dict(fail_search=99)),                             # E: tool always fails -> handoff
    ("F1", dict(note="I love flight QH118, please book exactly that one."), "FAILED:constraints", {}),   # F: user pushes a flight
    ("F2", dict(note="Just get me the cheapest flight of the day, whatever the time."), "DONE", {}),     # that breaks constraints
    #   F1: infeasible explicit wish -> must NOT substitute silently, ask the human.  F2: vague wish -> book a valid flight.
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=CORE, choices=PATTERNS)     # bonus: --patterns hybrid_replan ...
    ap.add_argument("--groups", nargs="+", default=list("ABCDEF"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()
    model, rows = make_model(), []
    model.invoke("ping")                                    # fail fast on a bad key / URL
    for cid, over, expect, extra in (c for c in CASES if c[0][0] in a.groups):
        for pattern in a.patterns:
            for _ in range(a.repeats):
                r = run(pattern, Constraints(**over), model=model, **extra)
                got = "DONE" if r["result"] == "DONE" else f"FAILED:{r['reason']}"
                rows.append(dict(case=cid, group=cid[0], pattern=pattern, expect=expect, got=got, success=int(got == expect),
                                 tool_calls=r["tool_calls"], denied=r["denied"], tokens=r["tokens"],
                                 latency=r["latency"], error=r["error"]))
                print(f"{cid} {pattern:13} expect={expect:18} got={got:18} calls={r['tool_calls']:2} denied={r['denied']} "
                      f"tokens={r['tokens']:5} {r['latency']:5.1f}s {r['error'] or ''}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, encoding="utf-8")
    g = df.groupby("pattern")
    handoff = df[df.expect != "DONE"].groupby("pattern").success.mean() * 100          # right handoff reason when it must fail
    summary = pd.DataFrame({"success%": g.success.mean() * 100, "handoff%": handoff, "tool_calls": g.tool_calls.mean(),
                            "denied": g.denied.mean(), "tokens": g.tokens.mean(), "latency_s": g.latency.mean(),
                            "errors": g.error.count()}).round(1)
    print("\n=== Summary (average per run) ===\n", summary.to_string())
    print("\n=== Success% by group ===\n", (df.pivot_table(index="group", columns="pattern", values="success") * 100).round(0).to_string())


if __name__ == "__main__":
    main()
