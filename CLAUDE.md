# BTVN#3 - Flight booking agent (LangChain / LangGraph)

University assignment (Agentic course). Deliverables: `.py` code + a report (`.md`).
Task: build a flight-booking agent with a **harness** layer, implement **3 design patterns**, and **evaluate** them.

## Requirements (from the assignment)
1. Harness layers: constraints as **data**, completion criteria checked by **code**, **permission** check, **handoff** to a human.
2. Same agent in 3 patterns: **ReAct**, **Plan-then-Execute**, **Hybrid**.
3. Evaluate the 3 patterns on the same benchmark.

## Files
| File | Role |
|---|---|
| `flight_agent.py` | Everything: mock tools, harness, 3 patterns + 1 bonus, `run()`, CLI. ~330 lines. |
| `evaluate.py` | Benchmark: 15 cases (groups A-F) x 3 patterns x `--repeats`; prints summary. `--out <csv>` optional. Bonus patterns via `--patterns`. |
| `test_harness.py` | Unit tests for the harness, no LLM needed (`pytest -q test_harness.py`). |
| `.env` | `NINEROUTER_API_KEY` (secret, never print/commit), `NINEROUTER_URL`, `NINEROUTER_MODEL`. Git-ignored. |
| `.env.example` | Template for `.env` with placeholder values. Committed. |
| `flight_agent copy.py`, `flight_agent_plan_then_execute.py`, `flight_agent_failure_mode.py` | User-supplied reference examples (ReAct, Plan-then-Execute, 4 failure modes). Style guide only; do not edit or import. |

## Design (flight_agent.py)
- **Constraints = data**: `Constraints` dataclass (`origin, destination, date, depart_before, max_price, note`). `to_prompt()` builds the user request; `is_ok(flight)` is the code check. `note` is free user text, not a constraint.
- **Permission = data**: `PERMISSION` table (`allow` / `ask` / `deny`; unknown tool = deny). `pay` is `ask` -> human approval via the `APPROVE` callback.
- **Harness**: `check()` (permission, loop guard = same call 3x, `book_seat` must satisfy constraints) -> `guarded()` (runs tool, empty result = error). **All 3 patterns call tools only through `guarded()`.** For `create_agent` it is wrapped as the `harness` middleware.
- **Done = code**: `is_done()` = exactly one paid booking, read back from `BOOKINGS`, satisfying constraints. The model's own claim is never trusted.
- **Handoff**: `handoff()` returns `reason` (`constraints` | `payment` | `tool_error`), `done_so_far`, `tried`, `question`.
- **Patterns**:
  - `react`: LangChain `create_agent` + harness middleware + `ModelCallLimitMiddleware(10)`.
  - `plan_execute`: code searches first, ONE structured-output call writes the plan (`Plan`/`Step`), code executes it, no replanning. Search error -> stop + handoff (known weakness, by design).
  - `hybrid_replan`: LLM plans once; code runs + observes each step. `error` -> retry that step in place (max 2, no LLM); still failing or `denied`/`not_found` -> replan (max 2, planner sees the error text, no re-search).
  - `hybrid`: fixed plan in code: find (ReAct, read-only tool, 1 recovery) -> select (code, cheapest valid) -> book -> pay (approval) -> verify (`is_done`). Plan is code, not an LLM call. Known weakness: select ignores free-text `note`.
  - `hybrid_replan` (bonus, BAOCAO Appendix A): LLM plans once; code runs + observes each step. Error -> retry in place (max 2, no LLM); still failing or `denied`/`not_found` -> replan (max 2). 100% benchmark.
- Module-level globals (`C`, `BOOKINGS`, `LOG`, `FAIL_SEARCH`, `APPROVE`) are reset by `run()`; runs must be sequential.

## Environment
- Windows 11, Python 3.13. Deps: `requirements.txt` (langchain, langgraph, langchain-openai, python-dotenv, pandas, pydantic, pytest).
- LLM via **9router** (OpenAI-compatible, `http://localhost:20128/v1`). Current model: `ag/gemini-3.8-flash-low` (~2x faster than `ag/gemini-3.8-flash`; the `ag/gemini-3.5-*` models are discontinued; `cx/*` hit usage limits).
- Router adds ~2000 tokens per LLM call, so absolute token counts are inflated; comparisons between patterns remain valid. Latency is noisy (5-75 s for the same work): average over repeats.
- Run: `python flight_agent.py [react|plan_execute|hybrid|hybrid_replan]` (asks for payment approval), `python evaluate.py --repeats 3`.
- Default model in code = `ag/gemini-3.8-flash-low` (same as benchmark). Setup: `cp .env.example .env`, then fill `NINEROUTER_API_KEY`.
- `load_dotenv()` needs a script file, not stdin; use `load_dotenv(".env")` in ad-hoc snippets.

## Benchmark cases (evaluate.py)
A simple, B many constraints, C impossible (expect handoff `constraints`), D human refuses payment (`payment`), E tool failure (E1 fails once -> recover; E2 always -> `tool_error`), F user pushes a violating flight (F1 expects handoff, must not silently substitute; F2 vague wish -> book a valid flight).
Success = outcome equals expected `DONE` / `FAILED:<reason>`. `handoff%` = success on the cases that must fail. `denied` = calls blocked by the harness.

## Findings (final, 15 cases x 3 reps, gemini-3.8-flash-low; numbers in BAOCAO.md)
- ReAct 100% (9.3k tokens/run); Plan-Execute 93.3% (E1 fails 3/3: search error stops the plan; 2.4k tokens); Hybrid 93.3% (F1 fails 3/3: books VJ606 instead of QH118, `note` ignored by select; 5.8k tokens).
- Benchmark `denied` comes from D1 (pay refused) and loop guard in E2. No violating `book_seat` was ever attempted, so harness blocking of bad bookings is shown only by unit tests.
- Bonus (BAOCAO Appendix A): Hybrid + Replan 100% (2.5k tokens).

## Status
- [x] Code, 13 unit tests pass, benchmark run3 done.
- [x] Report `BAOCAO.md` written (main 3 patterns; bonus in appendix).
- Run logs and CSVs are not kept in the repo. Regenerate with `python evaluate.py --repeats 3`.

## Conventions
- Keep code lean and close to the reference examples' style; no features beyond the requirements.
- Never read/print `.env` values; check only whether a key is set.
- Respond to the user in Vietnamese.
