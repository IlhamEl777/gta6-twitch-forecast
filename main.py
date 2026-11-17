"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Model: per-hour log-log regression on past big launches.
  For each forecast hour h = 2..24, fit log V_h = a_h + b_h * log V_1 over the case's
  own comparables (past launches' release days), where V_1 is the launch's hour-1
  print and V_h its print in hour h. The median forecast is the fitted value at
  this case's hour-1 print; the five quantiles are the median times exp(z_tau * s_h),
  s_h the residual standard deviation of the fit (z the standard normal quantiles).
  Only comparables whose release day is fully observed (all 24 hourly points) and
  whose hour-1 print is at least 10,000 viewers are used: hour-1 prints below that
  are pre-release trickle from launches that had not unlocked yet at 00:00 UTC, not
  a real midnight audience, and they would flatten the fit towards a constant level.
  The fit is done at run time from the case file itself, so the final case (which
  holds all 42 comparables) is scored with the full set.

  Why this shape: launch-day Twitch curves share one timezone-driven shape (a dip
  in hours 2-4 while only the earliest timezones are live, a jump at the US unlock,
  a broad evening peak), but the evening level is far from proportional to hour 1:
  across past launches the fitted slope falls from ~1.0 at hour 2 to ~0.5-0.6 in the
  evening, i.e. a small hour 1 means a much larger evening multiple than a big one.
  A persistence forecast (the house baseline) misses that ramp entirely.

  Practice (4 past launches, answers from SullyGnome hourly averages, pinball loss
  on the log scale, lower better): this model 0.213, house baseline 0.387.

Standard library only, deterministic (no randomness, sorted iteration, fixed
arithmetic), no network. Reads every *.in in the input directory and writes one
<id>.out per case. Fallbacks keep the output valid: if too few comparables qualify,
all fully-observed comparables are used; if none qualify, the house baseline
formula is used.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Spread of the house baseline (persistence), used only as a fallback.
BASELINE_SIGMA = 0.15
# Hour-1 prints below this are not a real midnight audience (launch not yet
# unlocked at 00:00 UTC); comparables below it are left out of the fit.
MIN_HOUR1 = 10000.0
# Need at least this many qualifying comparables, else relax the filter.
MIN_COMPARABLES = 8
# No hour's log-spread is trusted below this: with very few comparables the
# residual spread collapses towards zero, which would claim false certainty.
MIN_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as JavaScript's Math.round(x * 100) / 100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_hours(comparable: dict) -> dict[int, float] | None:
    """{hour 1..24: viewers} for the comparable's release day, or None if incomplete."""
    try:
        day_start = parse_ts(comparable["day_start"])
    except (KeyError, ValueError):
        return None
    points: dict[int, float] = {}
    for row in comparable.get("hourly") or []:
        try:
            when = parse_ts(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= when < day_start + timedelta(days=1) and viewers > 0:
            points[when.hour + 1] = viewers
    if len(points) != 24:
        return None
    return points


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, list[float]]:
    """Ordinary least squares y ~ a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def pstdev(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def clamp_quantiles(values: list[float]) -> list[float | int]:
    """Round, keep > 0 and <= 1e9, and non-decreasing."""
    out: list[float] = []
    prev = 0.0
    for v in values:
        q = min(max(v, 1.0), 1e9)
        q = max(q, prev)
        out.append(q)
        prev = q
    return [round2(q) for q in out]


def baseline_quantiles(p1: float, h: int) -> list[float | int]:
    """The house baseline (persistence + widening spread); fallback only."""
    raw = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z]
    return clamp_quantiles(raw)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    # The training set: this case's comparables with a complete release day.
    curves: list[dict[int, float]] = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if hours is not None:
            curves.append(hours)
    genuine = [c for c in curves if c[1] >= MIN_HOUR1]
    train = genuine if len(genuine) >= MIN_COMPARABLES else curves

    fits: dict[int, tuple[float, float]] = {}
    if len(train) >= 2:
        for h in range(2, 25):
            xs = [math.log(c[1]) for c in train]
            ys = [math.log(c[h]) for c in train]
            a, b, resid = ols(xs, ys)
            fits[h] = (a + b * log_p1, max(MIN_SIGMA, pstdev(resid)))

    rows = []
    for row in case["forecast"]:
        h = int(row["hour"])
        if h in fits:
            mu, sigma = fits[h]
            raw = [math.exp(mu + z * sigma) for z in Z]
            rows.append({"hour": h, "q": clamp_quantiles(raw)})
        else:
            rows.append({"hour": h, "q": baseline_quantiles(p1, h)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
