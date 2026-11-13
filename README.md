# polymarket-nvidia-ceo-forecast-01 entry — Falcon (w7)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

The fast-adapting variant:

- **Median = persistence.** The last observed price.
- **Short memory:** the last 720 minutes feed everything.
- **Median-based sigma:** 1.4826 × the median absolute one-minute
  change (floor 0.0008, cap 0.003).
- **Short anchor ladder:** 1, 3, 5, 15, 45, 90, 120 minutes,
  interpolated LINEARLY in h and held flat beyond the last anchor,
  instead of the family's sqrt(h) curve.
- **Spread = an even 50/50 blend** of the empirical quantiles and
  the Gaussian walk.
- **Volatility regime:** the last 12 hours vs the whole history,
  clamped to [0.8, 1.4].

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.352 against the
house baseline's 0.399 on the same folds — the family's best local
score.

## License

MIT — see `LICENSE`.
