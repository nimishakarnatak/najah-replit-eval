"""Load the conversation set: encrypted parts (data/*.jsonl.gz.enc, key in the NAJAH_DATA_KEY secret) or plain parts
(data/*.jsonl.gz, for a direct upload to Replit). Each line is one episode: episode_id, source ("rated" or "pool"),
module, language, turns, delivery_evidence, and rater_critical (true/false) for rated episodes."""
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_episodes():
    enc = sorted((ROOT / 'data').glob('*.jsonl.gz.enc')); plain = sorted((ROOT / 'data').glob('*.jsonl.gz'))
    if not enc and not plain:
        raise SystemExit('No conversation data in data/. Run prepare_data.py on the study computer, then commit the encrypted parts.')
    blobs = []
    if enc:
        key = os.environ.get('NAJAH_DATA_KEY')
        if not key: raise SystemExit('Encrypted data found but the NAJAH_DATA_KEY secret is not set (Replit: Tools > Secrets).')
        from cryptography.fernet import Fernet
        f = Fernet(key.strip().encode())
        blobs += [f.decrypt(p.read_bytes()) for p in enc]
    blobs += [p.read_bytes() for p in plain]
    episodes = {}
    for b in blobs:
        for line in gzip.decompress(b).decode('utf-8').splitlines():
            if line.strip():
                e = json.loads(line); episodes[e['episode_id']] = e
    return episodes
