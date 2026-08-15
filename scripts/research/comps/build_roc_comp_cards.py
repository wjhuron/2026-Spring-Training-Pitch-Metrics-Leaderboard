"""
Generate full-season pitcher cards for the 11 ROC pitchers + their top-3
same-hand pitch-mix comps, organized into ~/Downloads/ROC-Comps/<Last-First>/.

Strategy: one Cards.py run per team into a shared staging dir (cheap dedup for
comps that appear twice), then copy the 4 relevant cards into each ROC folder.
"""
import os, subprocess, shutil, sys
from collections import defaultdict

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'Downloads', 'ROC-Comps')
STAGING = os.path.join(HOME, 'Downloads', '_roc_card_staging')
CARDS = '/Users/wallyhuron/Huronalytics/Cards.py'

# ROC pitcher (Last, First) -> team is ROC for all
ROC = ['Champlain, Chandler','Cranz, Robert','Kent, Jackson','Kent, Zak',
       'Lara, Andry','Penrod, Zach','Perales, Luis','Sinclair, Jack',
       'Tolman, Erik','Yean, Eddy','Young, Luke']

# same-hand top-3 comps
COMPS = {
 'Champlain, Chandler':['Littell, Zack','Clarke, Taylor','Civale, Aaron'],
 'Cranz, Robert':['Johnson, Pierce','Bachar, Lake','Martinez, Nick'],
 'Kent, Jackson':['Gore, MacKenzie','Kikuchi, Yusei','Bubic, Kris'],
 'Kent, Zak':['Bradish, Kyle','Keller, Brad','Waldrep, Hurston'],
 'Lara, Andry':['Mikolas, Miles','Winn, Keaton','Ureña, Walbert'],
 'Penrod, Zach':['Rojas, Kendry','Holton, Tyler','Rodriguez, Eduardo'],
 'Perales, Luis':['Adams, Travis','Stanek, Ryne','Clarke, Taylor'],
 'Sinclair, Jack':['Morris, Kade','Petty, Chase','Perkins, Jack'],
 'Tolman, Erik':['Lynch IV, Daniel','Enns, Dietrich','Latz, Jacob'],
 'Yean, Eddy':['Bello, Brayan','Loáisiga, Jonathan','Lange, Alex'],
 'Young, Luke':['Bachar, Lake','Jameson, Drey','Brazobán, Huascar'],
}

# team tab for each MLB comp
TEAM = {
 'Adams, Travis':'MIN','Bachar, Lake':'MIA','Bello, Brayan':'BOS','Bradish, Kyle':'BAL',
 'Brazobán, Huascar':'NYM','Bubic, Kris':'KCR','Civale, Aaron':'ATH','Clarke, Taylor':'ARI',
 'Enns, Dietrich':'BAL','Gore, MacKenzie':'TEX','Holton, Tyler':'DET','Jameson, Drey':'ARI',
 'Johnson, Pierce':'CIN','Keller, Brad':'PHI','Kikuchi, Yusei':'LAA','Lange, Alex':'KCR',
 'Latz, Jacob':'TEX','Littell, Zack':'WSH','Loáisiga, Jonathan':'ARI','Lynch IV, Daniel':'KCR',
 'Martinez, Nick':'TBR','Mikolas, Miles':'WSH','Morris, Kade':'ATH','Perkins, Jack':'ATH',
 'Petty, Chase':'CIN','Rodriguez, Eduardo':'ARI','Rojas, Kendry':'MIN','Stanek, Ryne':'STL',
 'Ureña, Walbert':'LAA','Waldrep, Hurston':'ATL','Winn, Keaton':'SFG',
}
for n in ROC: TEAM[n] = 'ROC'

def slug(name):
    parts = name.split(', ')
    if len(parts) == 2:
        return f"Season-{(parts[0]+parts[1]).replace(' ','')}.png"
    return f"Season-{name.replace(' ','').replace(',','')}.png"

def folder(name):  # "Last, First" -> "Last-First"
    l, f = name.split(', ', 1)
    return f"{l}-{f}".replace(' ', '')

# Special handling — these arms are NOT plain single-team cards:
#   SPECIAL_AUTO      -> auto whole-season combine (traded within MLB -> 2TM)
#   SPECIAL_ALLLEVELS -> cross-level MLB+AAA combine (--all-levels)
# They are excluded from the per-team runs and generated separately below.
SPECIAL_AUTO = {'Civale, Aaron'}
SPECIAL_ALLLEVELS = {'Kent, Zak', 'Yean, Eddy'}
SPECIAL = SPECIAL_AUTO | SPECIAL_ALLLEVELS

# ---- 1. group unique standard pitchers by team (specials excluded) ----
by_team = defaultdict(set)
for n in ROC:
    if n not in SPECIAL: by_team['ROC'].add(n)
for lst in COMPS.values():
    for c in lst:
        if c not in SPECIAL: by_team[TEAM[c]].add(c)

shutil.rmtree(STAGING, ignore_errors=True)
os.makedirs(STAGING, exist_ok=True)

def run_cards(extra_args, label):
    r = subprocess.run(['python3', CARDS, '--start', 'none',
                        '--output-dir', STAGING] + extra_args,
                       cwd='/Users/wallyhuron/Huronalytics',
                       capture_output=True, text=True)
    tail = '\n'.join(r.stdout.strip().splitlines()[-4:])
    print(tail, flush=True)
    if r.returncode != 0:
        print(f"  STDERR ({label}):", r.stderr.strip()[-800:], flush=True)

# ---- 2. run Cards.py once per team (standard cards) ----
teams = sorted(by_team)
n_total = sum(len(v) for v in by_team.values()) + len(SPECIAL)
print(f"Generating {n_total} cards: {len(teams)} team runs + {len(SPECIAL)} special\n", flush=True)
for i, tm in enumerate(teams, 1):
    names = sorted(by_team[tm])
    print(f"[{i}/{len(teams)}] TEAM {tm}: {names}", flush=True)
    run_cards(['--team', tm, '--pitchers', ';'.join(names)], tm)

# ---- 2b. special cards ----
for nm in sorted(SPECIAL_AUTO):
    print(f"[special/auto] {nm} -> whole-season combine", flush=True)
    run_cards(['--pitchers', nm], nm)
if SPECIAL_ALLLEVELS:
    names = sorted(SPECIAL_ALLLEVELS)
    print(f"[special/all-levels] {names} -> MLB+AAA combine", flush=True)
    run_cards(['--all-levels', '--pitchers', ';'.join(names)], 'all-levels')

# ---- 3. copy into per-ROC-pitcher folders ----
print("\n=== Assembling folders ===", flush=True)
missing = []
for roc in ROC:
    fdir = os.path.join(BASE, folder(roc))
    os.makedirs(fdir, exist_ok=True)
    wanted = [roc] + COMPS[roc]
    for w in wanted:
        src = os.path.join(STAGING, slug(w))
        dst = os.path.join(fdir, slug(w))
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            missing.append((roc, w, slug(w)))
    have = len([f for f in os.listdir(fdir) if f.endswith('.png')])
    print(f"  {folder(roc)}: {have}/4 cards", flush=True)

if missing:
    print("\n!!! MISSING CARDS:", flush=True)
    for roc, w, s in missing:
        print(f"   {roc}  <-  {w}  ({s})", flush=True)
else:
    print("\nAll 44 card slots filled.", flush=True)
