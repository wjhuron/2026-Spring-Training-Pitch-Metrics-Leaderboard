# launchd jobs — current state

Three user LaunchAgents point at this repo:

| Job | Runs | Schedule |
|---|---|---|
| com.huronalytics.absdaily | `scripts/abs_daily.py` | 7:30 daily |
| com.huronalytics.refreshfg | `fg_overrides.py` (root stub) | 9:00 daily |
| com.huronalytics.autopull | `scripts/auto-pull.sh` | keep-alive |

## Pickle refresh is manual-only (policy, 2026-08-15)

The old com.huronalytics.refreshpickle job (7:00 daily) was removed at
Wally's direction: the local pickle refreshes only on manual runs.

```bash
python3 -m pipeline.refresh_pickle
```

(or `python3 refresh_pickle.py` via the root stub). HitterCards also
self-heals: it downloads the CI release pickle when the local one is
stale. History: the scheduled job had crashed on every run since install
(env var held a path where the client expected JSON content, and the repo
service_account.json 403s on a division workbook anyway); both defects
are fixed in code, but the schedule itself is retired.

## refreshfg

Still scheduled, runs through the root `fg_overrides.py` stub, so it
works unchanged after the 2026-08 reorg. To retire the stub, install the
replacement plist in this folder:

```bash
cp docs/launchd/com.huronalytics.refreshfg.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.huronalytics.refreshfg.plist
launchctl load ~/Library/LaunchAgents/com.huronalytics.refreshfg.plist
```

Then `fg_overrides.py` (root stub) can be deleted. `refresh_pickle.py`
(root stub) stays for manual runs, or use the `-m` form above and delete
it too.
