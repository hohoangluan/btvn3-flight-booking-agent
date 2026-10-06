"""Harness unit tests - no LLM needed. Run: pytest -q test_harness.py"""
import flight_agent as fa


def setup(approve=True, **over):
    fa.C, fa.FAIL_SEARCH, fa.APPROVE = fa.Constraints(**over), 0, (lambda args: approve)
    fa.BOOKINGS.clear(), fa.LOG.clear()


def test_constraints_are_data():
    setup()
    ok = {f["flight"]: fa.C.is_ok(f) for f in fa.FLIGHTS}
    assert ok["VN122"] and ok["VJ606"]
    assert not ok["VJ604"] and not ok["QH118"] and not ok["VN210"]      # price / afternoon / other route


def test_booking_that_breaks_constraints_is_denied():
    setup()
    assert fa.guarded("book_seat", {"flight": "QH118"})["status"] == "denied" and not fa.BOOKINGS


def test_payment_needs_human_approval():
    setup(approve=False)
    code = fa.guarded("book_seat", {"flight": "VN122"})["code"]
    assert fa.guarded("pay", {"code": code})["status"] == "denied" and not fa.BOOKINGS[code]["paid"]
    fa.APPROVE = lambda args: True
    assert fa.guarded("pay", {"code": code})["status"] == "ok"


def test_done_is_checked_by_code():
    setup()
    assert not fa.is_done()
    code = fa.guarded("book_seat", {"flight": "VN122"})["code"]
    assert not fa.is_done()                                              # held but not paid
    fa.guarded("pay", {"code": code})
    assert fa.is_done()


def test_two_paid_bookings_is_not_done():
    setup()
    for flight in ("VN122", "VJ606"):
        fa.guarded("pay", {"code": fa.guarded("book_seat", {"flight": flight})["code"]})
    assert not fa.is_done()


def test_loop_guard_blocks_4th_identical_call():
    setup()
    args = {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"}
    assert [fa.guarded("search_flights", args)["status"] for _ in range(4)] == ["ok", "ok", "ok", "denied"]


def test_empty_result_and_bad_args_are_errors():
    setup()
    fa.TOOLS["empty"], fa.PERMISSION["empty"] = (lambda: {}), "allow"
    assert fa.guarded("empty", {})["status"] == "error"
    assert fa.guarded("search_flights", {"wrong": 1})["status"] == "error"
    del fa.TOOLS["empty"], fa.PERMISSION["empty"]


def test_unknown_tool_denied_and_no_prompt_for_missing_booking():
    prompts = []
    setup()
    fa.APPROVE = lambda args: prompts.append(args) or False
    assert fa.guarded("cancel_booking", {"code": "x"})["status"] == "denied"
    assert fa.guarded("pay", {"code": "NOPE"})["status"] == "not_found" and not prompts   # human never asked


def test_handoff_reason_matches_failure():
    setup()
    assert fa.handoff()["reason"] == "constraints"
    fa.FAIL_SEARCH = 1
    fa.guarded("search_flights", {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"})
    assert fa.handoff()["reason"] == "tool_error"
    setup(approve=False)
    fa.guarded("pay", {"code": fa.guarded("book_seat", {"flight": "VN122"})["code"]})
    assert fa.handoff()["reason"] == "payment"


class FakeModel:
    """Stands in for the LLM: each planner call returns the next scripted plan, and the prompt it saw is kept."""
    def __init__(self, *plans):
        self.plans, self.prompts = list(plans), []

    def with_structured_output(self, schema, **kw):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda p: self.prompts.append(p.to_string()) or fa.Plan(steps=[fa.Step(**s) for s in self.plans.pop(0)]))


GOOD = [dict(tool="book_seat", flight="VN122"), dict(tool="pay", code="$booking_code")]


def test_replan_fixes_a_tool_error_in_place_without_replanning():
    setup()
    fa.FAIL_SEARCH = 1                                                   # search fails once
    model = FakeModel(GOOD)
    fa.hybrid_replan(model, {})
    assert fa.is_done() and len(model.prompts) == 1                      # fixed by a retry: ONE plan, no replan
    assert [r["status"] for t, _, r in fa.LOG if t == "search_flights"] == ["error", "ok"]


def test_replan_gives_up_when_the_fix_does_not_work():
    setup()
    fa.FAIL_SEARCH = 99
    model = FakeModel()
    fa.hybrid_replan(model, {})
    assert not fa.is_done() and not model.prompts and fa.handoff()["reason"] == "tool_error"


def test_replan_sees_why_a_step_was_refused_and_does_not_search_again():
    setup()
    model = FakeModel([dict(tool="book_seat", flight="QH118")], GOOD)    # 1st plan breaks the constraints
    fa.hybrid_replan(model, {})
    assert fa.is_done() and len(model.prompts) == 2
    assert "breaks the constraints" in model.prompts[1]                  # the planner was told the reason
    assert sum(t == "search_flights" for t, _, _ in fa.LOG) == 1         # no wasted search


def test_replan_stops_when_the_human_refuses_to_pay():
    setup(approve=False)
    model = FakeModel(GOOD, GOOD)
    fa.hybrid_replan(model, {})
    assert not fa.is_done() and len(model.prompts) == 1 and fa.handoff()["reason"] == "payment"

