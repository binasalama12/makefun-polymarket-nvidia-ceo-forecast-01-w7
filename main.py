#!/usr/bin/env python3
"""Falcon forecaster for Polymarket's "Next CEO of NVIDIA?" event
(make.fun polymarket-nvidia-ceo-forecast-01, wallet w7 variant).

Reads one case from the input directory: every market of the event
with its Yes price at each minute up to the checkpoint. Writes, for
each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

The Falcon variant is built for speed of adaptation:

- Short memory: volatility and the empirical change distribution
  come from the last 720 minutes only.
- Sigma is median-based (1.4826 * median absolute one-minute
  change), floor 0.0008, cap 0.003, so single prints cannot blow
  out the fan.
- The anchor set stops at 120 minutes (1, 3, 5, 15, 45, 90, 120):
  beyond the last anchor the empirical row is held flat instead of
  extrapolated, and horizons between anchors are interpolated
  LINEARLY in h rather than in sqrt(h) — with a short anchor ladder
  the sqrt curve over-bends the mid horizons.
- The blend is even, 50/50 empirical/Gaussian.
- The regime factor uses a 12-hour recent window against the whole
  history, clamped to the narrow [0.8, 1.4].

The median forecast is the last observed price. Quantiles are
clamped to [0, 1], kept non-decreasing and rounded to 6 decimals.
Deterministic, standard library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZNORM = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

DEFAULT_SIGMA = 0.002
HISTORY = 720            # minutes of history for vol estimates
ANCHORS = [1, 3, 5, 15, 45, 90, 120]
EMP_WEIGHT = 0.5
SIGMA_FLOOR = 0.0008
SIGMA_CAP = 0.003
MIN_EMP_HISTORY = 300


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def round6(v):
    return round(clamp01(v) * 1e6) / 1e6 + 0.0


def sorted_quantile(vals, tau):
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def abs_steps(prices):
    return [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]


def mean_of(vals):
    return sum(vals) / len(vals) if vals else 0.0


def median_of(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def anchor_quantiles(hist):
    table = {}
    n = len(hist)
    for h in ANCHORS:
        if h >= n:
            continue
        changes = sorted(hist[i + h] - hist[i] for i in range(n - h))
        table[h] = [sorted_quantile(changes, tau) for tau in LEVELS]
    return table


def empirical_at(table, h):
    """Empirical change quantiles at horizon h, linear in h between
    anchors, flat beyond the last one."""
    keys = sorted(table)
    if not keys:
        return None
    if h <= keys[0]:
        return table[keys[0]]
    if h >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            f = (h - lo) / (hi - lo)
            return [table[lo][j] * (1.0 - f) + table[hi][j] * f
                    for j in range(len(LEVELS))]
    return table[keys[-1]]


def market_sigma(prices):
    window = prices[-HISTORY:] if len(prices) > HISTORY else prices
    med = median_of(abs_steps(window))
    return min(max(1.4826 * med, SIGMA_FLOOR), SIGMA_CAP)


def regime_scale(prices):
    """Last 12 h of movement vs the whole history, clamp [0.8, 1.4]."""
    if len(prices) <= 1440:
        return 1.0
    overall = mean_of(abs_steps(prices))
    if overall <= 1e-9:
        return 1.0
    recent = mean_of(abs_steps(prices[-721:]))
    return min(1.4, max(0.8, recent / overall))


def forecast_market(prices, horizon, sigma):
    x = prices[-1]
    window = prices[-HISTORY:] if len(prices) > HISTORY else list(prices)
    table = anchor_quantiles(window) if len(window) >= MIN_EMP_HISTORY else {}
    scale = regime_scale(prices)
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [x + z * sigma * sq for z in ZNORM]
        emp = empirical_at(table, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_WEIGHT * (x + emp[j]) + (1.0 - EMP_WEIGHT) * gauss[j]
                   for j in range(len(LEVELS))]
        rows.append(sorted(round6(x + (v - x) * scale) for v in row))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != LEVELS:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {LEVELS}")
        horizon = int(case["horizon_minutes"])
        parsed = []
        for market in case["markets"]:
            prices = [float(p) for p in market["prices"]]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            parsed.append((market["id"], prices))
        sigmas = {mid: market_sigma(prices) for mid, prices in parsed
                  if len(prices) >= MIN_EMP_HISTORY}
        fallback_sigma = median_of(list(sigmas.values())) if sigmas else DEFAULT_SIGMA
        forecasts = []
        for mid, prices in parsed:
            forecasts.append({"id": mid,
                              "q": forecast_market(prices, horizon,
                                                   sigmas.get(mid, fallback_sigma))})
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / f"{case_id}.out").write_text(out)


if __name__ == "__main__":
    main()
