# launchd jobs — current state and optional migration

Four user LaunchAgents point at this repo:

| Job | Runs | Status |
|---|---|---|
| com.huronalytics.refreshpickle | `refresh_pickle.py` (root stub) | 7:00 daily |
| com.huronalytics.refreshfg | `fg_overrides.py` (root stub) | 9:00 daily |
| com.huronalytics.absdaily | `scripts/abs_daily.py` | 7:30 daily |
| com.huronalytics.autopull | `scripts/auto-pull.sh` | keep-alive |

After the 2026-08 reorg, the first two root files are thin stubs that call
`pipeline.refresh_pickle` and `pipeline.fg_overrides`. The stubs keep the
old plists working. Nothing breaks if you never touch this.

To retire the stubs, install the replacement plists in this folder:

```bash
cp docs/launchd/com.huronalytics.refreshpickle.plist ~/Library/LaunchAgents/
cp docs/launchd/com.huronalytics.refreshfg.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.huronalytics.refreshpickle.plist
launchctl load ~/Library/LaunchAgents/com.huronalytics.refreshpickle.plist
launchctl unload ~/Library/LaunchAgents/com.huronalytics.refreshfg.plist
launchctl load ~/Library/LaunchAgents/com.huronalytics.refreshfg.plist
```

Then delete `refresh_pickle.py` and `fg_overrides.py` from the repo root.

IMPORTANT (2026-08-15): the old refreshpickle plist exported
`GOOGLE_SERVICE_ACCOUNT_JSON=<path to repo service_account.json>`. Two
problems, both fixed:

1. The client treated the env var as inline JSON, so a path crashed it.
   The 7:00 job had failed on every run since it was installed. The
   client now accepts a path or inline JSON.
2. Even as a path, that credential (leaderboard-reader@st-leaderboard)
   gets 403 on at least one division workbook. The default gspread
   credential in `~/.config/gspread/service_account.json` is the one with
   full access, so the replacement plist drops the env var entirely.
