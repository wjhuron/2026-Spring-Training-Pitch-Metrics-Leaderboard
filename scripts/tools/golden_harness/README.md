# Golden-output harness (built 2026-08-15 for the cleanup branch)

Proves a pipeline change is behavior-neutral: replay a frozen Sheets
snapshot through `pipeline.process_data.main()` with the network layer
monkeypatched, and require byte-identical output against a baseline.

Usage (from any working directory; state lives NEXT TO these scripts, so
copy the folder somewhere disposable first — prestate/ + runs/ grow to a
few GB):

```bash
python3 capture_inputs.py                    # once: freeze Sheets rows (2m30s)
PYTHONHASHSEED=0 python3 golden_run.py prestate   # snapshot data/, js/, index.html (~3s)
PYTHONHASHSEED=0 python3 golden_run.py run BASE   # baseline (5m45s)
# ... make your code change ...
PYTHONHASHSEED=0 python3 golden_run.py run CHANGE # 5m45s
PYTHONHASHSEED=0 python3 golden_run.py compare BASE CHANGE
```

Budget ~15 min for a full verification. Timings measured 2026-08-17 on
583,619 frozen rows (pipeline itself 334s; the rest is snapshot/restore).

What is frozen: Sheets rows (both reads), FanGraphs guts + park factors,
the boxscore refresh window (pure cache serve), PYTHONHASHSEED. The
comparator masks generatedAt and ?v= stamps and ignores the input-side
caches (boxscore, milb boxscore, game weather). Every run restores the
files it touched from prestate, so runs are repeatable back to back.

Two restore gaps, measured 2026-08-17. WATCH covers `data`, `js`, and
`index.html`, but the pipeline also stamps `trade.html` and `catch.html`,
so a run rolls their `?v=` back and leaves it there — check `git status`
after a run. And files over COPY_LIMIT (100 MB) are hashed, not copied,
so they cannot be restored; `data/all_pitches_rs_cache.pkl` (340 MB) is
currently the only one.

Verified 2026-08-15: two baseline runs on 574,614 frozen rows were
byte-identical; the whole repo reorg then reproduced the baseline exactly.

Protocol notes: re-capture inputs and re-baseline in the SAME session as
any verification (day-boundary cache drift is real); never compare runs
made on different days.
