#!/usr/bin/env python3
"""Support agent on Mistral's Responses API (public preview, Oct 2026).

Demos, in the order the video uses them:
    python agent.py --demo hello       # one call, print the text
    python agent.py --demo calls       # ask a question, show the function_call items
    python agent.py --demo agent       # the full loop until the model answers
    python agent.py --demo reasoning   # same, with reasoning effort high, reasoning items shown
    python agent.py --demo ticket      # adds the create_ticket tool; prints the ticket JSON
    python agent.py --demo count       # count input tokens before sending
    python agent.py --demo agent --model mistral-small-latest
    python agent.py --demo compare --models mistral-large-4,mistral-small-latest \
        --price mistral-large-4=IN,OUT --price mistral-small-latest=IN,OUT

Facts this code relies on (docs + mistralai 3.6.0 types, checked 2026-10-29):
  - The API is stateless: the app resends the whole history in `input` each call.
  - Output items: type "message" (content parts "output_text"), "function_call"
    (name, arguments as a JSON *string*, call_id), "reasoning" (summary / content).
  - Tool results go back as {"type": "function_call_output", "call_id", "output": str}.
  - One response can hold several function calls (parallel_tool_calls defaults to true).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from mistralai.client import Mistral  # docs and SDK 3.6.0 import path

from tools_data import FUNCS, TICKET_TOOL, TOOLS

DEFAULT_MODEL = "mistral-large-4"
QUESTION = "Where is order 104, and can I still return it?"
TICKET_QUESTION = (
    "A customer asks about order 104: where is it and can they still return it? "
    "Find out, then file a ticket for the support team."
)


def client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        sys.exit("MISTRAL_API_KEY is not set")
    return Mistral(api_key=key)


# ---------- helpers for the three item types ----------

def message_text(response) -> str:
    """Join the output_text parts of every message item."""
    parts = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    parts.append(part.text)
    return "\n".join(parts)


def reasoning_text(response) -> list[str]:
    """Collect whatever reasoning the model exposed: summary and/or content parts.

    Which of these is filled depends on the model (docs: "plaintext reasoning,
    depending on the model"). encrypted_content is opaque and not shown.
    """
    out = []
    for item in response.output:
        if item.type != "reasoning":
            continue
        for s in item.summary or []:
            out.append(f"[summary] {s.text}")
        for c in item.content or []:
            text = getattr(c, "text", None)
            if text:
                out.append(f"[content] {text}")
        if item.encrypted_content and not out:
            out.append("[encrypted reasoning present, not readable]")
    return out


def function_calls(response):
    return [i for i in response.output if i.type == "function_call"]


def as_input_items(response) -> list[dict]:
    """Turn typed output items back into dicts the next request accepts."""
    # Do not add exclude_unset=True: the SDK serializer then drops the "type" field.
    return [i.model_dump(exclude_none=True, by_alias=True) for i in response.output]


# ---------- the agent loop ----------

def run_agent(cl: Mistral, model: str, question: str, tools: list[dict],
              reasoning: dict | None = None, show_reasoning: bool = False,
              max_rounds: int = 6, usage: dict | None = None):
    """Run the loop until the model answers without tool calls.

    `usage`, when given, accumulates input/output tokens across all calls.
    """
    history: list[dict] = [{"role": "user", "content": question}]
    last = None
    for round_no in range(1, max_rounds + 1):
        kwargs = {"model": model, "input": history, "tools": tools, "max_output_tokens": 4096}
        if reasoning:
            kwargs["reasoning"] = reasoning
        last = cl.responses.create(**kwargs)
        if usage is not None:
            usage["calls"] = usage.get("calls", 0) + 1
            # SDK 3.6.0 leaves usage unset if the server's usage shape doesn't match
            # its model; record that instead of reporting zero tokens.
            if last.usage:
                usage["input"] = usage.get("input", 0) + (last.usage.input_tokens or 0)
                usage["output"] = usage.get("output", 0) + (last.usage.output_tokens or 0)
            else:
                usage["missing"] = True

        if show_reasoning:
            for line in reasoning_text(last):
                print(f"  {line}")

        history.extend(as_input_items(last))      # keep the model's own items in the history
        calls = function_calls(last)
        if not calls:
            return last, history

        for call in calls:
            args = json.loads(call.arguments or "{}")  # arguments arrive as a JSON string
            print(f"  -> {call.name}({args})")
            result = FUNCS[call.name](**args)
            history.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),       # output is a string
            })
    return last, history


# ---------- demos ----------

def demo_hello(cl, model):
    # Reasoning tokens count against max_output_tokens: Large 4 reasons by default
    # and spent ~400 of them on this one sentence (checked 2026-10-29).
    r = cl.responses.create(model=model, input="Write a one-sentence greeting for a bookshop.",
                            max_output_tokens=1024)
    for item in r.output:
        print(f"[{item.type}]")
    print(message_text(r))
    if r.status != "completed":
        print(f"status: {r.status} ({r.incomplete_details})")
    if r.usage:
        print(f"output tokens: {r.usage.output_tokens}, of which reasoning: "
              f"{r.usage.output_tokens_details.reasoning_tokens}")


def demo_calls(cl, model):
    r = cl.responses.create(model=model, input=QUESTION, tools=TOOLS, max_output_tokens=1024)
    for item in r.output:
        print(json.dumps(item.model_dump(exclude_none=True), indent=2))


def demo_agent(cl, model, reasoning=None, show_reasoning=False):
    t0 = time.perf_counter()
    r, _ = run_agent(cl, model, QUESTION, TOOLS, reasoning=reasoning, show_reasoning=show_reasoning)
    print(message_text(r))
    print(f"[{model}: {time.perf_counter() - t0:.1f}s]")


def demo_ticket(cl, model):
    tools = TOOLS + [TICKET_TOOL]
    _, history = run_agent(cl, model, TICKET_QUESTION, tools)
    tickets = [h for h in history if h.get("type") == "function_call_output"
               and '"ticket_id"' in h["output"]]
    for t in tickets:
        print(json.dumps(json.loads(t["output"]), indent=2))
    if not tickets:
        print("No ticket filed; see the final message above.")


def demo_compare(cl, models: list[str], prices: dict[str, tuple[float, float]]):
    """Same agent, several models: wall time, tokens, and cost when prices are given.

    Prices are per million input/output tokens, passed on the command line
    (--price model=in,out). Nothing is hard-coded: read them from
    mistral.ai/pricing on the day you record and show the page on screen.
    """
    rows = []
    for model in models:
        usage: dict = {}
        t0 = time.perf_counter()
        r, _ = run_agent(cl, model, QUESTION, TOOLS, usage=usage)
        secs = time.perf_counter() - t0
        cost = None
        tin = tout = None
        if not usage.get("missing"):
            tin, tout = usage.get("input", 0), usage.get("output", 0)
            if model in prices:
                pin, pout = prices[model]
                cost = tin / 1e6 * pin + tout / 1e6 * pout
        first = (message_text(r).strip().splitlines() or ["(no text answer)"])[0][:70]
        rows.append((model, secs, usage.get("calls", 0), tin, tout, cost, first))
    print(f"{'model':<24}{'secs':>7}{'calls':>7}{'in tok':>9}{'out tok':>9}{'cost USD':>10}  answer")
    for model, secs, calls, tin, tout, cost, first in rows:
        c = f"{cost:.5f}" if cost is not None else "n/a"
        tin = "n/a" if tin is None else tin
        tout = "n/a" if tout is None else tout
        print(f"{model:<24}{secs:>7.1f}{calls:>7}{tin:>9}{tout:>9}{c:>10}  {first}")


def demo_count(cl, model):
    history = [{"role": "user", "content": QUESTION}]
    r = cl.responses.count_input_tokens(model=model, input=history, tools=TOOLS)
    print(json.dumps(r.model_dump(exclude_none=True), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True,
                    choices=["hello", "calls", "agent", "reasoning", "ticket", "count", "compare"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high",
                    help="reasoning effort for --demo reasoning (model-dependent)")
    ap.add_argument("--models", default="mistral-large-4,mistral-small-latest",
                    help="comma-separated model IDs for --demo compare")
    ap.add_argument("--price", action="append", default=[],
                    help="model=input_usd_per_M,output_usd_per_M (repeatable, for --demo compare)")
    a = ap.parse_args()
    cl = client()
    if a.demo == "compare":
        prices = {}
        for p in a.price:
            name, _, pair = p.partition("=")
            pin, pout = (float(x) for x in pair.split(","))
            prices[name] = (pin, pout)
        demo_compare(cl, [s.strip() for s in a.models.split(",") if s.strip()], prices)
        return
    if a.demo == "hello":
        demo_hello(cl, a.model)
    elif a.demo == "calls":
        demo_calls(cl, a.model)
    elif a.demo == "agent":
        demo_agent(cl, a.model)
    elif a.demo == "reasoning":
        demo_agent(cl, a.model, reasoning={"effort": a.effort}, show_reasoning=True)
    elif a.demo == "ticket":
        demo_ticket(cl, a.model)
    elif a.demo == "count":
        demo_count(cl, a.model)


if __name__ == "__main__":
    main()
