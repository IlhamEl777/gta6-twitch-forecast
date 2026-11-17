# gta6-twitch-forecast

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day (2026-11-19 UTC), from the first hour's viewer count — entry
for make.fun competition `gta6-twitch-ccu-forecast-01`.

## Program

`main.py` (Python 3.13, standard library only, deterministic, no network) reads
every `*.in` case in the input directory and writes one `<id>.out` forecast per
case: five quantiles (5/25/50/75/95%) of Twitch concurrent viewers for hours 2–24.

## Model

Per-hour log-log regression on past big launches. For each hour h, the program
fits `log V_h = a_h + b_h · log V_1` over the case's own comparables (past
launches' release days, each with all 24 hourly points and an hour-1 print of at
least 10,000 viewers), then forecasts the median at this case's hour-1 print with
quantiles from the fit's residual spread. Launch-day curves share one
timezone-driven shape — a dip in hours 2–4, a jump at the US unlock, a broad
evening peak — but the evening multiple shrinks as hour 1 grows (fitted slope
falls from ~1.0 at hour 2 to ~0.5–0.6 in the evening), which a persistence
forecast misses entirely.

Practice (4 past launches, pinball loss on the log scale, lower is better):
this model **0.213** vs house persistence baseline **0.387**.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
