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
        --price mistral-large-4=IN,OUT --price mistral-small-latest=IN,OUT --runs 3
    python agent.py --demo regions --models zai-glm-5-3-flash,deepseek-v4-1-flash-1017
    python agent.py --demo longctx --model deepseek-v4-1-flash-1017   # needs make_log.py first
    python agent.py --demo agent --server eu    # EU endpoint (api.eu.mistral.ai)

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
import statistics
import sys
import time

from mistralai.client import Mistral  # docs and SDK 3.6.0 import path
from mistralai.client import errors

from make_log import ROOT_CAUSES
from tools_data import FUNCS, TICKET_TOOL, TOOLS

DEFAULT_MODEL = "mistral-large-4"
QUESTION = "Where is order 104, and can I still return it?"
TICKET_QUESTION = (
    "A customer asks about order 104: where is it and can they still return it? "
    "Find out, then file a ticket for the support team."
)


def client(server: str = "global") -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        sys.exit("MISTRAL_API_KEY is not set")
    return Mistral(api_key=key, server=server)


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
        details = r.usage.output_tokens_details
        print(f"output tokens: {r.usage.output_tokens}, of which reasoning: "
              f"{details.reasoning_tokens if details else 'not reported'}")


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


def demo_compare(cl, models: list[str], prices: dict[str, tuple[float, float]], runs: int = 1):
    """Same agent, several models: wall time, tokens, and cost when prices are given.

    Prices are per million input/output tokens, passed on the command line
    (--price model=in,out). Nothing is hard-coded: read them from
    mistral.ai/pricing on the day you record and show the page on screen.
    With --runs N each model runs N times and the row shows the median.
    An API error (e.g. 429 on a capacity-limited preview model) becomes the row's answer.
    """
    rows = []
    for model in models:
        secs_l, tin_l, tout_l, calls, first = [], [], [], 0, ""
        for _ in range(runs):
            usage: dict = {}
            t0 = time.perf_counter()
            try:
                r, _ = run_agent(cl, model, QUESTION, TOOLS, usage=usage)
            except errors.SDKError as e:
                first = f"HTTP {e.status_code}: {str(e)[:50]}"
                break
            secs_l.append(time.perf_counter() - t0)
            calls = usage.get("calls", 0)
            if not usage.get("missing"):
                tin_l.append(usage.get("input", 0))
                tout_l.append(usage.get("output", 0))
            first = (message_text(r).strip().splitlines() or ["(no text answer)"])[0][:70]
        secs = statistics.median(secs_l) if secs_l else float("nan")
        tin = int(statistics.median(tin_l)) if tin_l else None
        tout = int(statistics.median(tout_l)) if tout_l else None
        cost = None
        if tin is not None and model in prices:
            pin, pout = prices[model]
            cost = tin / 1e6 * pin + tout / 1e6 * pout
        rows.append((model, secs, calls, tin, tout, cost, first))
    if runs > 1:
        print(f"(median of {runs} runs per model)")
    print(f"{'model':<26}{'secs':>7}{'calls':>7}{'in tok':>9}{'out tok':>9}{'cost USD':>10}  answer")
    for model, secs, calls, tin, tout, cost, first in rows:
        c = f"{cost:.5f}" if cost is not None else "n/a"
        tin = "n/a" if tin is None else tin
        tout = "n/a" if tout is None else tout
        print(f"{model:<26}{secs:>7.1f}{calls:>7}{tin:>9}{tout:>9}{c:>10}  {first}")


def demo_regions(models: list[str]):
    """Which endpoints list which models. Docs: regional endpoints only serve
    models hosted in that region; the global endpoint commits to no location."""
    found = {}
    for server in ("global", "eu", "us"):
        try:
            ids = {m.id for m in client(server).models.list().data}
        except errors.SDKError as e:
            ids = None
            print(f"{server}: HTTP {e.status_code}")
        found[server] = ids
    print(f"{'model':<28}{'global':>8}{'eu':>6}{'us':>6}")
    for model in models:
        cells = ["?" if found[s] is None else ("yes" if model in found[s] else "no")
                 for s in ("global", "eu", "us")]
        print(f"{model:<28}{cells[0]:>8}{cells[1]:>6}{cells[2]:>6}")


def demo_longctx(cl, model, path="logs/incident.log"):
    """Paste a long log, ask for the three root causes, check which planted causes it names."""
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run python make_log.py first")
    log = open(path).read()
    prompt = ("Below is an application log from one night. Find the three independent root "
              "causes of the errors. For each: one line with the cause and the first timestamp "
              "where it shows. Ignore noise.\n\n" + log)
    history = [{"role": "user", "content": prompt}]
    n = cl.responses.count_input_tokens(model=model, input=history)
    print(f"input: {n.model_dump().get('input_tokens', n)} tokens")
    t0 = time.perf_counter()
    # 16k output: Large 4 reasons by default and spent all of 4096 before answering (checked 2026-10-29)
    r = cl.responses.create(model=model, input=history, max_output_tokens=16384)
    secs = time.perf_counter() - t0
    answer = message_text(r)
    print(answer)
    if r.status != "completed":
        print(f"status: {r.status} ({r.incomplete_details})")
    low = answer.lower()
    hits = [name for name, words in ROOT_CAUSES.items() if any(w in low for w in words)]
    print(f"[{model}: {secs:.1f}s, found {len(hits)}/3: {', '.join(hits) or 'none'}]")
    if r.usage:
        details = r.usage.output_tokens_details
        print(f"[tokens in {r.usage.input_tokens}, out {r.usage.output_tokens}, reasoning "
              f"{details.reasoning_tokens if details else 'not reported'}]")


def demo_count(cl, model):
    history = [{"role": "user", "content": QUESTION}]
    r = cl.responses.count_input_tokens(model=model, input=history, tools=TOOLS)
    print(json.dumps(r.model_dump(exclude_none=True), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True,
                    choices=["hello", "calls", "agent", "reasoning", "ticket", "count", "compare",
                             "regions", "longctx"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high",
                    help="reasoning effort for --demo reasoning (model-dependent)")
    ap.add_argument("--models", default="mistral-large-4,mistral-small-latest",
                    help="comma-separated model IDs for --demo compare / regions")
    ap.add_argument("--runs", type=int, default=1, help="runs per model for --demo compare (median)")
    ap.add_argument("--server", default="global", choices=["global", "eu", "us"],
                    help="API endpoint: global (no location commitment), eu or us")
    ap.add_argument("--price", action="append", default=[],
                    help="model=input_usd_per_M,output_usd_per_M (repeatable, for --demo compare)")
    a = ap.parse_args()
    if a.demo == "regions":
        demo_regions([s.strip() for s in a.models.split(",") if s.strip()])
        return
    cl = client(a.server)
    if a.demo == "compare":
        prices = {}
        for p in a.price:
            name, _, pair = p.partition("=")
            pin, pout = (float(x) for x in pair.split(","))
            prices[name] = (pin, pout)
        demo_compare(cl, [s.strip() for s in a.models.split(",") if s.strip()], prices, a.runs)
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
    elif a.demo == "longctx":
        demo_longctx(cl, a.model)


if __name__ == "__main__":
    main()
