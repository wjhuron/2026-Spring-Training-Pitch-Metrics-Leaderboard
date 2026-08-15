"""drive_pickle.py — move pickle artifacts through Google Drive privately.

Replaces the public GitHub release as the transport for:
  - all_pitches_rs_cache.pkl  (rebuilt by CI every run; HitterCards pulls it)
  - _pitches2025_training.pkl (static prior-season training set)

Files are owned by the huronalytics service account (the same one the six
division workbooks use), so both CI (GOOGLE_SERVICE_ACCOUNT_JSON secret) and
local runs (~/.config/gspread/service_account.json) can read/write them and
nothing is public.

Usage:
    python3 scripts/builders/drive_pickle.py upload <local_path> [--name NAME]
    python3 scripts/builders/drive_pickle.py download <name_or_id> <local_path>
    python3 scripts/builders/drive_pickle.py list
"""
import os, sys, json, mimetypes

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DRIVE = 'https://www.googleapis.com/drive/v3'
UPLOAD = 'https://www.googleapis.com/upload/drive/v3'


def _session():
    from google.auth.transport.requests import AuthorizedSession
    sa_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    from google.oauth2.service_account import Credentials
    scopes = ['https://www.googleapis.com/auth/drive']
    if sa_json:
        creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=scopes)
    else:
        path = os.path.expanduser('~/.config/gspread/service_account.json')
        if not os.path.exists(path):
            path = os.path.join(ROOT, 'service_account.json')
        creds = Credentials.from_service_account_file(path, scopes=scopes)
    return AuthorizedSession(creds)


def find(s, name):
    q = f"name='{name}' and trashed=false"
    r = s.get(f'{DRIVE}/files', params={'q': q, 'fields': 'files(id,name,size,modifiedTime)'})
    r.raise_for_status()
    files = r.json().get('files', [])
    return files[0] if files else None


def upload(path, name=None):
    s = _session()
    name = name or os.path.basename(path)
    size = os.path.getsize(path)
    existing = find(s, name)
    meta = {'name': name}
    if existing:
        init = s.patch(f"{UPLOAD}/files/{existing['id']}?uploadType=resumable",
                       json=meta, headers={'X-Upload-Content-Length': str(size)})
    else:
        init = s.post(f'{UPLOAD}/files?uploadType=resumable',
                      json=meta, headers={'X-Upload-Content-Length': str(size)})
    init.raise_for_status()
    loc = init.headers['Location']
    CHUNK = 64 * 1024 * 1024  # 64MB, multiple of 256KB
    with open(path, 'rb') as f:
        sent = 0
        while sent < size:
            chunk = f.read(CHUNK)
            end = sent + len(chunk) - 1
            r = s.put(loc, data=chunk,
                      headers={'Content-Range': f'bytes {sent}-{end}/{size}'})
            if r.status_code not in (200, 201, 308):
                raise RuntimeError(f'upload failed: {r.status_code} {r.text[:200]}')
            sent += len(chunk)
            print(f'  uploaded {sent}/{size} bytes', flush=True)
    fid = (r.json().get('id') if r.status_code in (200, 201) else existing['id'])
    print(f'uploaded {name} -> fileId {fid}')
    return fid


def download(name_or_id, path):
    s = _session()
    fid = name_or_id
    if not name_or_id.replace('-', '').replace('_', '').isalnum() or len(name_or_id) < 20:
        f = find(s, name_or_id)
        if not f:
            raise SystemExit(f'no Drive file named {name_or_id}')
        fid = f['id']
    r = s.get(f'{DRIVE}/files/{fid}?alt=media', stream=True)
    r.raise_for_status()
    n = 0
    with open(path, 'wb') as out:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            out.write(chunk)
            n += len(chunk)
    print(f'downloaded {n / 1048576:.1f} MB -> {path}')


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == 'upload':
        name = None
        if '--name' in sys.argv:
            name = sys.argv[sys.argv.index('--name') + 1]
        upload(sys.argv[2], name)
    elif cmd == 'download':
        download(sys.argv[2], sys.argv[3])
    elif cmd == 'list':
        s = _session()
        r = s.get(f'{DRIVE}/files', params={'fields': 'files(id,name,size,modifiedTime)'})
        for f in r.json().get('files', []):
            print(f"{f['name']:40s} {int(f.get('size', 0))/1048576:8.1f} MB  {f['modifiedTime']}  {f['id']}")
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
