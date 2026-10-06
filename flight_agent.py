"""BTVN#3 - Flight booking agent: ONE harness, THREE patterns (ReAct / Plan-then-Execute / Hybrid).

    Agent = Model + Harness

Harness (plain Python, the model cannot skip it):
    1. Constraints are DATA     -> Constraints class, checked by check()
    2. Permission check         -> check(): booking must meet constraints, pay needs the human's OK
    3. Done is checked by CODE  -> is_done(), reads the bookings back
    4. Handoff to a human       -> handoff(), when the agent fails
    + loop guard (same call 3x) and "empty result = error" (from flight_agent_failure_mode.py)

Patterns (same model, tools, harness):
    react         the model decides ONE step at a time (LangChain create_agent)
    plan_execute  the model writes the WHOLE plan once, then code runs it - no replanning
    hybrid        fixed plan in code: find (ReAct) -> select (code) -> book -> pay -> verify (code)
    hybrid_replan LLM plans -> code runs + observes each step: an error is fixed in place (retry), else replan (max 2)

LLM = 9router, set in .env: NINEROUTER_API_KEY, NINEROUTER_URL, NINEROUTER_MODEL
Run:  python flight_agent.py [react|plan_execute|hybrid|hybrid_replan]
"""
import json
import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


# =====================================================================
# 1. CONSTRAINTS ARE DATA - the prompt is built FROM it, the harness CHECKS it in code
# =====================================================================
@dataclass
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"      # morning flight
    max_price: int = 2_000_000        # VND
    note: str = ""                    # free text from the user (NOT a constraint, may even push against them)

    def to_prompt(self) -> str:
        return (f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
                f"departing before {self.depart_before}, price at most {self.max_price:,} VND. {self.note}").strip()

    def is_ok(self, f: dict) -> bool:
        """Does this flight (or booking) satisfy ALL constraints?"""
        return ((f["origin"], f["destination"]) == (self.origin, self.destination)
                and f["depart"].startswith(self.date)
                and f["depart"][11:16] < self.depart_before
                and f["price"] <= self.max_price)


# =====================================================================
# MOCKUP TOOLS - fake data, no network
# =====================================================================
def _f(flight, origin, destination, depart, price):
    return dict(flight=flight, origin=origin, destination=destination, depart=depart, price=price)


FLIGHTS = [
    _f("VN122", "SGN", "DAD", "2026-10-07T08:10", 1_850_000),
    _f("VJ604", "SGN", "DAD", "2026-10-07T08:30", 2_480_000),   # morning but too expensive
    _f("VJ606", "SGN", "DAD", "2026-10-07T10:30", 1_350_000),
    _f("QH118", "SGN", "DAD", "2026-10-07T15:40", 1_640_000),   # cheap but afternoon
    _f("VN210", "SGN", "HAN", "2026-10-07T06:30", 2_300_000),
    _f("VJ150", "SGN", "HAN", "2026-10-07T09:45", 1_450_000),
    _f("VN212", "SGN", "HAN", "2026-10-07T11:20", 1_900_000),
    _f("QH220", "SGN", "HAN", "2026-10-07T13:00", 1_100_000),
    _f("VN301", "HAN", "SGN", "2026-10-08T07:00", 1_700_000),
    _f("VJ302", "HAN", "SGN", "2026-10-08T18:30", 1_200_000),
]
C = Constraints()               # the current request (set by run)
BOOKINGS = {}                   # our fake booking database
LOG = []                        # every tool call: (tool name, args, result)
FAIL_SEARCH = 0                 # fault injection: the next N searches time out
APPROVE = lambda args: True     # the human's decision on a payment (set by run)
PERMISSION = {"search_flights": "allow", "get_booking": "allow", "book_seat": "allow", "pay": "ask"}   # data; unknown tool = deny


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    global FAIL_SEARCH
    if FAIL_SEARCH > 0:
        FAIL_SEARCH -= 1
        return {"status": "error", "error": "service timeout"}
    return {"status": "ok", "flights": [f for f in FLIGHTS if f["origin"] == origin.upper()
                                        and f["destination"] == destination.upper() and f["depart"].startswith(date)]}


def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
    info = next((f for f in FLIGHTS if f["flight"] == flight), None)
    if info is None:
        return {"status": "not_found", "flight": flight}
    code = f"{flight}-12A"
    BOOKINGS[code] = {"code": code, "paid": False, **info}
    return {"status": "ok", **BOOKINGS[code]}


def pay(code: str) -> dict:
    """Pay for a held booking. This spends money and cannot be undone."""
    if code not in BOOKINGS:
        return {"status": "not_found", "code": code}
    BOOKINGS[code]["paid"] = True
    return {"status": "ok", **BOOKINGS[code]}


def get_booking(code: str) -> dict:
    """Read a booking back from the system."""
    return {"status": "ok", **BOOKINGS[code]} if code in BOOKINGS else {"status": "not_found"}


TOOLS = {t.__name__: t for t in [search_flights, book_seat, pay, get_booking]}


# =====================================================================
# THE HARNESS
# =====================================================================
# ---- 1+2. CONSTRAINT + PERMISSION CHECK: runs BEFORE the tool, using the constraint DATA
def check(name: str, args: dict) -> str | None:
    """Return None if allowed, or the reason why the call must be blocked."""
    perm = PERMISSION.get(name, "deny")
    if perm == "deny":
        return f"{name} is not permitted"
    if sum(1 for t, a, _ in LOG if (t, a) == (name, args)) >= 3:
        return "the same call was already made 3 times (loop)"
    if name == "book_seat":
        f = next((f for f in FLIGHTS if f["flight"] == args.get("flight")), None)
        if f is None or not C.is_ok(f):
            return f"{args.get('flight')} breaks the constraints: {C.to_prompt()}"
    if perm == "ask" and args.get("code") in BOOKINGS and not APPROVE(args):
        return "the user did not approve this payment"
    return None


def guarded(name: str, args: dict) -> dict:
    """Run a tool through the harness. ALL three patterns call tools only via this function."""
    reason = check(name, args)
    if reason:                                              # blocked: the tool does NOT run
        result = {"status": "denied", "reason": reason}
    else:
        try:
            result = TOOLS[name](**args) or {"status": "error", "error": "empty result"}   # {} is NOT "nothing found"
        except Exception as e:                              # bad arguments
            result = {"status": "error", "error": str(e)}
    LOG.append((name, args, result))
    return result


@wrap_tool_call
def harness(request, handler):
    """LangChain middleware: every tool call of a create_agent agent goes through guarded()."""
    call = request.tool_call
    return ToolMessage(content=json.dumps(guarded(call["name"], call["args"])), tool_call_id=call["id"], name=call["name"])


# ---- 3. DONE IS CHECKED BY CODE: never trust the model saying "I'm done"
def is_done() -> bool:
    """Done = exactly one paid booking, read back from the system, that satisfies the constraints."""
    paid = [b for b in map(get_booking, BOOKINGS) if b["paid"]]
    return len(paid) == 1 and C.is_ok(paid[0])


# ---- 4. HANDOFF: when the agent fails, give a human 3 things
def handoff() -> dict:
    if any(t == "pay" and r["status"] == "denied" for t, _, r in LOG):
        reason, question = "payment", "The payment was not approved. Do you want to approve it?"
    elif any(r["status"] == "error" for _, _, r in LOG):
        reason, question = "tool_error", "A tool failed (service error). Retry later?"
    else:
        reason, question = "constraints", "No flight meets all constraints. Which one can we relax: departure time or maximum price?"
    return {"reason": reason, "done_so_far": [f"{c}: paid={b['paid']}" for c, b in BOOKINGS.items()] or ["Nothing booked, nothing paid"],
            "tried": [f"{t}({a}) -> {r['status']}" for t, a, r in LOG],
            "question": question}


# =====================================================================
# THE THREE PATTERNS - each takes (model, config) and only acts through the harness
# =====================================================================
SYSTEM = ("You are a flight booking agent. Use the tools: search_flights, book_seat, pay. "
          "Only book a flight that meets ALL constraints. If a tool fails, retry. "
          "If no flight meets the constraints, stop and say so.")


def limit():
    return ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")     # budget: max 10 model calls


def ask():
    return {"messages": [{"role": "user", "content": C.to_prompt()}]}


# ---- 1. ReAct: reason -> act -> observe, one step at a time
def react(model, cfg):
    agent = create_agent(model=model, tools=list(TOOLS.values()), system_prompt=SYSTEM, middleware=[harness, limit()])
    agent.invoke(ask(), cfg)


# ---- 2. Plan-then-Execute: ONE model call writes the whole plan, then plain code runs it
class Step(BaseModel):
    tool: str          # one of: book_seat, pay, get_booking
    flight: str = ""   # for book_seat, e.g. "VN122"
    code: str = ""     # for pay / get_booking: write "$booking_code" (filled in from book_seat)


class Plan(BaseModel):
    steps: list[Step]  # an EMPTY list means: "no flight meets the constraints"


PLANNER = ChatPromptTemplate.from_messages([
    ("system", "You plan a flight booking. Write the WHOLE plan at once, as a list of steps. "
               "Tools: book_seat(flight), pay(code), get_booking(code). "
               "The booking code is not known yet: write \"$booking_code\" as code and it will be filled in. "
               "Only book a flight that meets ALL constraints. If none does, return an empty plan."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}\nSo far: {so_far}"),
])


def plan_execute(model, cfg):
    found = guarded("search_flights", dict(origin=C.origin, destination=C.destination, date=C.date))
    if found["status"] != "ok":
        return                                              # no flight list -> cannot plan
    plan = (PLANNER | model.with_structured_output(Plan, method="function_calling")).invoke(
        {"goal": C.to_prompt(), "flights": json.dumps(found["flights"]), "so_far": "nothing yet"}, cfg)
    for step in plan.steps:                                 # EXECUTE: no model call from here on
        code = next((r["code"] for t, _, r in reversed(LOG) if t == "book_seat" and r["status"] == "ok"), "")
        args = {k: v for k, v in (("flight", step.flight), ("code", code if step.code == "$booking_code" else step.code)) if v}
        if guarded(step.tool, args)["status"] != "ok":
            break                                           # the plan cannot adapt -> stop


# ---- 3. Hybrid: fixed plan in code. Only "find" is ReAct; select/book/pay/verify are code.
def hybrid(model, cfg):
    finder = create_agent(model=model, tools=[search_flights], middleware=[harness, limit()],
                          system_prompt="Find flights for the request with search_flights. If it errors, retry. "
                                        "Do NOT book anything. Reply DONE when you have the flight list.")
    for _ in range(2):                                      # find (ReAct) + one recovery if the search never worked
        finder.invoke(ask(), cfg)
        if any(t == "search_flights" and r["status"] == "ok" for t, _, r in LOG):
            break
    flights = [f for t, _, r in LOG if t == "search_flights" and r["status"] == "ok" for f in r["flights"]]
    good = sorted((f for f in flights if C.is_ok(f)), key=lambda f: f["price"])      # select: code
    if good:
        booking = guarded("book_seat", {"flight": good[0]["flight"]})                # book
        if booking["status"] == "ok":
            guarded("pay", {"code": booking["code"]})                                # pay (needs approval)
    # verify: is_done() is called by run()


# ---- 4. Hybrid + replan: LLM plans -> code runs the steps and OBSERVES each result
#         one error  -> FIX that step right there (retry it, no model call)
#         too many   -> a fix did not work, or the step was refused -> REPLAN (LLM sees WHY it failed)
def hybrid_replan(model, cfg, max_fix=2, max_replan=2):
    planner = PLANNER | model.with_structured_output(Plan, method="function_calling")

    def act(tool, args):                                    # one step through the harness + FIX: a tool error is retried here
        for _ in range(max_fix + 1):
            result = guarded(tool, args)
            if result["status"] != "error":                 # ok, or denied/not_found: retrying cannot change those
                break
        return result

    found = act("search_flights", dict(origin=C.origin, destination=C.destination, date=C.date))
    if found["status"] != "ok":
        return                                              # no flight list even after the fixes -> handoff
    flights = found["flights"]
    for _ in range(max_replan + 1):                         # 1 plan + up to max_replan replans
        plan = planner.invoke({"goal": C.to_prompt(), "flights": json.dumps(flights),
                               "so_far": json.dumps([f"{t}({a}) -> {r['status']} {r.get('error') or r.get('reason') or ''}".strip()
                                                     for t, a, r in LOG])}, cfg)    # the planner sees WHY a step failed
        for s in plan.steps:                                # execute + observe step by step, no model call
            code = next((r["code"] for t, _, r in reversed(LOG) if t == "book_seat" and r["status"] == "ok"), "")
            args = {k: v for k, v in (("flight", s.flight), ("code", code if s.code == "$booking_code" else s.code)) if v}
            if act(s.tool, args)["status"] != "ok":
                break                                       # fix failed or step refused -> replan with the error in LOG
        else:
            return                                          # whole plan ran (or empty plan = no flight fits)
        if any(t == "pay" and r["status"] == "denied" for t, _, r in LOG):
            return                                          # the human said no: replanning cannot change that


PATTERNS = {"react": react, "plan_execute": plan_execute, "hybrid": hybrid,      # the 3 required patterns
            "hybrid_replan": hybrid_replan}                                     # bonus (BAOCAO Phụ lục A)


# =====================================================================
# RUNNER
# =====================================================================
def make_model():
    return ChatOpenAI(base_url=os.getenv("NINEROUTER_URL", "http://localhost:20128/v1"),
                      api_key=os.environ["NINEROUTER_API_KEY"],
                      model=os.getenv("NINEROUTER_MODEL", "ag/gemini-3.8-flash-low"), temperature=0)


def run(pattern, constraints=None, approve=lambda args: True, fail_search=0, model=None) -> dict:
    """Run one pattern on one request. The HARNESS decides the result, not the model."""
    global C, APPROVE, FAIL_SEARCH
    C, APPROVE, FAIL_SEARCH = constraints or Constraints(), approve, fail_search
    BOOKINGS.clear(), LOG.clear()
    cb, err, t0 = UsageMetadataCallbackHandler(), None, time.perf_counter()
    try:
        PATTERNS[pattern](model or make_model(), {"callbacks": [cb]})
    except Exception as e:
        err = repr(e)
    done = is_done()
    ho = None if done else handoff()
    return {"result": "DONE" if done else "FAILED", "reason": ho and ho["reason"], "error": err,
            "tool_calls": len(LOG), "denied": sum(r["status"] == "denied" for _, _, r in LOG),
            "tokens": sum(u.get("total_tokens", 0) for u in cb.usage_metadata.values()),
            "latency": round(time.perf_counter() - t0, 1),
            "booking": list(BOOKINGS.values()) if done else None, "handoff": ho}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pattern = sys.argv[1] if len(sys.argv) > 1 else "react"
    print(f"=== pattern: {pattern} ===\nUser's request: {Constraints().to_prompt()}")
    out = run(pattern, approve=lambda a: input(f"\nHuman: approve payment {a}? (y/n): ").strip().lower() == "y")
    print("\n--- trace ---")
    for i, (t, a, r) in enumerate(LOG, 1):
        print(f"[{i}] {t}({a}) -> {r['status']} {r.get('reason', '')}")
    print("\n--- result ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))
