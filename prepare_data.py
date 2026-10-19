"""Build the conversation data for Replit. Run on the study computer only, by the data owner (no API calls).

Reads the study project (najah-llm-judge): the 300 rated episodes from its split-2 database, with the rater label
(critical = either primary rater flagged a critical failure), and the 3,263-episode unrated pool
(data/challenge_pool_episodes.json, automated de-identification; manual privacy review pending).
Keeps only what the judge receives (module, language, turns without the English translation, delivery evidence) plus
episode_id, source and the rated label. Participant codes, treatment, gender, rater data and weights are dropped.
Self-check: for every episode, the request text built by najah_eval equals the study run's request text.

Writes data/conversations-NNN.jsonl.gz.enc (500 episodes per part), encrypted with a key kept OUTSIDE the repo in
--key-file (created on first run, reused afterwards). Put that key in the Replit Secret NAJAH_DATA_KEY. With --plain it
writes unencrypted data/conversations-NNN.jsonl.gz instead (git-ignored; for a direct upload to Replit, not GitHub).

Usage: .venv/bin/python prepare_data.py [--project ~/Desktop/najah/najah-llm-judge] [--key-file ~/Desktop/najah/najah-data-key.txt] [--plain]
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from najah_eval.screen import load_config, build_request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=str(Path.home() / 'Desktop/najah/najah-llm-judge'))
    ap.add_argument('--key-file', default=str(Path.home() / 'Desktop/najah/najah-data-key.txt'))
    ap.add_argument('--part-size', type=int, default=500); ap.add_argument('--plain', action='store_true'); a = ap.parse_args()
    proj = Path(a.project).expanduser()
    sys.path[:0] = [str(proj), str(proj / 'scripts')]
    from najah.store import Store
    from najah.judges import SYSTEM
    from reference import critical_cases
    from run_extra import input_data as study_input_data
    from run_critical_variants import prompt_text, FORMAT
    cfg, prompts = load_config()
    if (prompts['system'], prompts['rubric'], prompts['format']) != (SYSTEM, prompt_text('B'), FORMAT):
        raise SystemExit('Prompt files differ from the study project; not exporting.')
    s = Store(str(proj / 'data/study-split2.sqlite')); crit = set(critical_cases(s))
    rated = s.episodes()
    pool = json.loads((proj / 'data/challenge_pool_episodes.json').read_text(encoding='utf-8'))['episodes']
    records = []
    for src, eps in (('rated', rated), ('pool', pool)):
        for e in eps:
            x = study_input_data(e)
            r = {'episode_id': e['episode_id'], 'source': src, 'module': x['module'], 'language': x['language'],
                 'turns': x['turns'], 'delivery_evidence': x['delivery_evidence']}
            if src == 'rated': r['rater_critical'] = e['episode_id'] in crit
            # self-check: identical request text to the study run (rubric + episode JSON, then the answer format)
            study_shared = json.dumps({'rubric': prompt_text('B'), 'episode': study_input_data(e)}, ensure_ascii=False)
            req = build_request(cfg, prompts, r)
            if req['messages'][0]['content'][0]['text'] != study_shared or req['system'] != SYSTEM:
                raise SystemExit(f"Request mismatch for {e['episode_id']}; not exporting.")
            records.append(r)
    ids = [r['episode_id'] for r in records]
    if len(ids) != len(set(ids)): raise SystemExit('Duplicate episode ids between rated and pool; not exporting.')
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    for old in list(out.glob('conversations-*.jsonl.gz*')): old.unlink()
    fernet = None
    if not a.plain:
        from cryptography.fernet import Fernet
        kf = Path(a.key_file).expanduser()
        if not kf.exists():
            kf.write_text(Fernet.generate_key().decode() + '\n'); os.chmod(kf, 0o600)
        fernet = Fernet(kf.read_text().strip().encode())
    parts = []
    for i in range(0, len(records), a.part_size):
        blob = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records[i:i + a.part_size]).encode('utf-8'))
        name = f'conversations-{i // a.part_size + 1:03d}.jsonl.gz' + ('' if a.plain else '.enc')
        (out / name).write_bytes(fernet.encrypt(blob) if fernet else blob); parts.append(name)
    size = sum((out / p).stat().st_size for p in parts) / 1e6
    print(f"Exported {len(records)} episodes: rated {len(rated)} ({len(crit)} critical), pool {len(pool)}. Request self-check passed for all.")
    print(f"Wrote {len(parts)} parts in data/ ({size:.1f} MB): {', '.join(parts)}")
    if fernet:
        print(f'Encrypted with the key in {Path(a.key_file).expanduser()} (outside the repo). Copy that key into the Replit Secret NAJAH_DATA_KEY.')
    else:
        print('Plain (unencrypted) parts: upload them straight to Replit; they are git-ignored and must not go to GitHub.')
    print('Reminder: the pool has automated de-identification only (manual privacy review pending).')

if __name__ == '__main__':
    main()
