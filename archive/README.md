# archive/ — superseded root-level modules

Archived 2026-07-29 during the full-codebase audit. Nothing in the live
tree imports these; git history has every version, this directory just
keeps them greppable.

- `_prebubble_render.py` — snapshot of Cards.py from before the
  percentile-bubble redesign (2026-07-13). Cards.py is the live renderer.
- `generate_batch_cards.py` — pre-Cards.py batch card generator (Jul 1);
  Cards.py absorbed the batch path.
- `generate_pitcher_card.py` — v30 single-card generator (Jul 1), output
  path hardcoded to one March game; superseded by Cards.py.
- `test_pipeline.py` — pre-embed-split smoke test; replaced by
  `scripts/validate_output.py` (which CI now runs).
- `stuff_plus_v10/` — Stuff+ v10 trainer + artifacts; superseded by
  `stuff_plus_v11/` (the only version CI runs).
