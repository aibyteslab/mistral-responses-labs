#!/usr/bin/env python3
"""Write a fake application log (about 40k tokens by default) with three planted root causes.

Everything is invented (example.com hosts, made-up services). The seed is fixed,
so every run writes the same file and every model gets the same input.

    python make_log.py            # writes logs/incident.log
    python make_log.py --lines 4000

The three causes the long-context demo checks for (ROOT_CAUSES below):
  1. the DB connection pool is exhausted after the 02:14 deploy
  2. the queue disk fills up and the workers crash
  3. the payments upstream's TLS certificate has expired, so checkout gets 502s
Everything else is noise, plus one harmless red herring (a deprecated-config warning).
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta

# keyword groups: an answer "finds" a cause if it mentions any word in the group
ROOT_CAUSES = {
    "db pool exhausted": ["pool", "connection pool"],
    "queue disk full": ["disk", "no space", "enospc"],
    "expired TLS cert": ["certificate", "tls", "cert"],
}

SERVICES = ["api-gateway", "orders", "catalog", "search", "auth", "worker", "checkout"]
PATHS = ["/v1/orders", "/v1/catalog/items", "/v1/search", "/v1/auth/refresh", "/v1/cart", "/health"]


def noise(rng: random.Random, ts: datetime) -> str:
    svc = rng.choice(SERVICES)
    kind = rng.random()
    if kind < 0.6:
        path = rng.choice(PATHS)
        return (f"{ts:%Y-%m-%dT%H:%M:%S}.{rng.randint(0, 999):03d}Z INFO  {svc} "
                f"req_id={rng.getrandbits(48):012x} GET {path} 200 {rng.randint(3, 180)}ms")
    if kind < 0.8:
        return (f"{ts:%Y-%m-%dT%H:%M:%S}.{rng.randint(0, 999):03d}Z DEBUG {svc} "
                f"cache hit ratio={rng.uniform(0.82, 0.97):.2f} keys={rng.randint(9000, 12000)}")
    if kind < 0.93:
        return (f"{ts:%Y-%m-%dT%H:%M:%S}.{rng.randint(0, 999):03d}Z INFO  {svc} "
                f"gc pause {rng.randint(2, 40)}ms heap={rng.randint(300, 900)}MB")
    # red herring: loud but harmless
    return (f"{ts:%Y-%m-%dT%H:%M:%S}.{rng.randint(0, 999):03d}Z WARN  {svc} "
            f"config key 'retry.legacy_backoff' is deprecated and ignored")


def planted(rng: random.Random, ts: datetime, minute: int, state: dict) -> str | None:
    """Return a symptom line for whichever cause is active at this minute, or None."""
    t = f"{ts:%Y-%m-%dT%H:%M:%S}.{rng.randint(0, 999):03d}Z"
    # cause 1: deploy at 02:14 lowers the pool size (logged once); timeouts from 02:20
    if minute >= 134 and not state.get("deployed"):
        state["deployed"] = True
        return f"{t} INFO  orders deploy v2.41.0 started by ci (config: db.pool.max_size 50 -> 5)"
    if minute >= 140 and rng.random() < 0.04:
        return (f"{t} ERROR orders db: timeout acquiring connection from pool after 30000ms "
                f"(active=5 idle=0 waiting={rng.randint(20, 90)})")
    # cause 2: queue volume fills from 03:00, workers crash
    if 180 <= minute and rng.random() < 0.02:
        return f"{t} WARN  worker /var/lib/queue usage {min(100, 80 + (minute - 180) // 3)}%"
    if minute >= 230 and rng.random() < 0.025:
        return (f"{t} ERROR worker write segment failed: [Errno 28] No space left on device; "
                f"worker exiting (code 1), restart {rng.randint(1, 40)}")
    # cause 3: upstream cert expires at 04:00 UTC
    if minute >= 240 and rng.random() < 0.03:
        return (f"{t} ERROR checkout upstream payments.example.com: SSL: CERTIFICATE_VERIFY_FAILED "
                f"certificate has expired (notAfter=2026-10-31T04:00:00Z) -> 502")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=1600, help="default is about 40k tokens")
    ap.add_argument("--out", default="logs/incident.log")
    a = ap.parse_args()
    rng = random.Random(7)
    state: dict = {}
    start = datetime(2026, 10, 31, 0, 0, 0)
    step = timedelta(seconds=6 * 3600 / a.lines)   # spread the lines over 6 hours
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for i in range(a.lines):
            ts = start + step * i
            minute = int((ts - start).total_seconds() // 60)
            f.write((planted(rng, ts, minute, state) or noise(rng, ts)) + "\n")
    print(f"wrote {a.out}: {a.lines} lines, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
