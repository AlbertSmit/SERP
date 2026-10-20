# Changelog

## SubConPack 2.0

- Use positive RSJ information weights, an explicit unknown-token prior, and
  English/Dutch substantive-token filtering.
- Normalize by live mean chunk length and score header, prefix, stem, and
  compound matches explicitly.
- Raise candidate pooling to `max(floor(0.4M), 25)` and add an abstaining
  trigram fallback for vaults of at most 100 chunks. Larger vaults skip trigram
  comparisons entirely.
- Gate packing by weighted coverage with a shaped rare-identifier exception.
- Stop at 8% residual demand, reject zero-gain candidates, and apply a 0.26
  first-marginal dropoff.
- Reuse term-match results within each packing call across scoring, coverage,
  marginal gain, and residual decay.
- Expose `min_coverage` and `rel_dropoff` on `packContext` and `fullPipeline`
  for development evaluation.
