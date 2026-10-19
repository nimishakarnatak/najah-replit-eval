"""Najah critical-failure screen on Replit.

  python run.py status                      data, credentials, calibration and progress (no API calls)
  python run.py test-call                   one paid call to check credentials and the model (about $0.03)
  python run.py calibrate [--max-usd 15]    score the 300 rated episodes, refit the cut-off to the team target
  python run.py screen [--max-usd 150]      score every remaining episode with the calibrated cut-off
  python run.py report                      write outputs/critical-screen-flags.csv and a summary (no API calls)

Results are appended to outputs/results.jsonl, so an interrupted run resumes where it stopped (failed episodes are
retried on the next run). outputs/ holds model reasons that quote conversations: keep it out of git (see .gitignore).
"""
import argparse
import csv
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from najah_eval.screen import load_config, client, score
from najah_eval.data import load_episodes

RESULTS = ROOT / 'outputs/results.jsonl'
CALIBRATION = ROOT / 'outputs/calibration.json'
SEED = 20261018

def cost(cfg, usage):
    p = cfg['price_per_million_tokens']
    return ((usage or {}).get('input_tokens', 0) * p['input'] + (usage or {}).get('output_tokens', 0) * p['output']) / 1e6

def load_results(cfg):
    ok, spent = {}, 0.0
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['model'] != cfg['model']: continue
            spent += cost(cfg, r.get('usage'))
            if r['status'] == 'ok': ok[r['episode_id']] = r
    return ok, spent

def run_jobs(cfg, prompts, episodes, ids, max_usd):
    done, spent = load_results(cfg)
    todo = [e for e in ids if e not in done]
    if not todo: print('Nothing to score.'); return
    cl, how = client()
    print(f"Scoring {len(todo)} episodes with {cfg['model']} ({how}); spent so far ${spent:.2f}; stop at ${max_usd:.2f}.", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    lock = threading.Lock(); state = {'spent': spent, 'n': 0, 'errors': 0, 'stopped': False}; t0 = time.time()
    def job(eid):
        with lock:
            if state['spent'] >= max_usd: state['stopped'] = True; return
        rec = {'episode_id': eid, 'source': episodes[eid]['source'], 'model': cfg['model'], 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
        try:
            rec.update(status='ok', **score(cl, cfg, prompts, episodes[eid]))
        except ValueError as exc:
            msg, usage = (exc.args + ({},))[:2]
            rec.update(status='error', error=str(msg)[:300], usage=usage if isinstance(usage, dict) else {})
        except Exception as exc:  # provider/network error after SDK retries; no conversation text is logged
            rec.update(status='error', error=type(exc).__name__, usage={})
        with lock:
            with open(RESULTS, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            state['spent'] += cost(cfg, rec.get('usage')); state['n'] += 1; state['errors'] += rec['status'] != 'ok'
            if state['n'] % 25 == 0 or state['n'] == len(todo):
                print(f"  {state['n']}/{len(todo)} done, {state['errors']} errors, spent ${state['spent']:.2f}, {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as ex: list(ex.map(job, todo))
    if state['stopped']: print(f'Stopped at the ${max_usd:.2f} budget; rerun with a higher --max-usd to continue.')
    if state['errors']: print(f"{state['errors']} episodes failed; rerun the same command to retry them.")

def curve(scores, labels, cut):
    pos = [e for e in scores if labels[e]]; neg = [e for e in scores if not labels[e]]
    c = sum(scores[e] >= cut for e in pos); fa = sum(scores[e] >= cut for e in neg)
    return c, len(pos), fa, len(neg)

def fit(scores, labels, max_fa):
    """Cut-off with the most critical cases caught while false flags stay within max_fa (ties: fewer false flags)."""
    best = None
    for t in sorted(set(scores.values())):
        c, _, fa, n = curve(scores, labels, t)
        if fa / n <= max_fa and (best is None or (c, -fa) > (best[1], -best[2])): best = (t, c, fa)
    return best[0] if best else None

def cmd_status(cfg, prompts, a):
    import os
    eps = load_episodes(); done, spent = load_results(cfg)
    print(f"Episodes: {len(eps)} ({', '.join(f'{k} {v}' for k, v in Counter(e['source'] for e in eps.values()).items())}); "
          f"rated critical cases: {sum(bool(e.get('rater_critical')) for e in eps.values())}")
    creds = 'Replit-managed' if os.environ.get('AI_INTEGRATIONS_ANTHROPIC_BASE_URL') else 'own key' if os.environ.get('ANTHROPIC_API_KEY') else 'NONE'
    print(f"Model: {cfg['model']}; credentials: {creds}; scored: {len(done)}/{len(eps)}; spent: ${spent:.2f}")
    print('Calibration: ' + (CALIBRATION.read_text() if CALIBRATION.exists() else 'not done yet (run: python run.py calibrate)'))

def cmd_test_call(cfg, prompts, a):
    eps = load_episodes(); eid = sorted(e for e in eps if eps[e]['source'] == 'rated')[0]
    cl, how = client(); r = score(cl, cfg, prompts, eps[eid])
    print(f"OK via {how}: {eid} -> critical {r['judgments']['critical']}, probability {r['judgments']['probability']}, "
          f"returned model {r['returned_model']}, cost ${cost(cfg, r['usage']):.3f} (not saved)")

def cmd_calibrate(cfg, prompts, a):
    eps = load_episodes()
    rated = sorted(e for e, x in eps.items() if x['source'] == 'rated' and x.get('rater_critical') is not None)
    run_jobs(cfg, prompts, eps, rated, a.max_usd)
    done, _ = load_results(cfg)
    scores = {e: done[e]['judgments']['probability'] for e in rated if e in done}; labels = {e: bool(eps[e]['rater_critical']) for e in scores}
    if len(scores) < len(rated): print(f'Only {len(scores)}/{len(rated)} rated episodes scored; rerun calibrate before fitting.'); return
    max_fa = cfg['target']['max_false_flags']; cut = fit(scores, labels, max_fa)
    c, P, fa, N = curve(scores, labels, cut)
    print(f'\nCut-off for {cfg["model"]}: probability >= {cut}: caught {c}/{P} ({c / P:.0%}), false flags {fa}/{N} ({fa / N:.0%}), flagged {(c + fa) / len(scores):.0%}')
    print('Nearby cut-offs:')
    for t in sorted(set(scores.values())):
        cc, _, ff, _ = curve(scores, labels, t)
        if 0.03 <= ff / N <= 0.40: print(f'  >= {t:>3}: caught {cc}/{P} ({cc / P:.0%}), false flags {ff / N:.0%}')
    # honest estimate: 5-fold cross-validation of the cut-off choice
    pos = [e for e in scores if labels[e]]; neg = [e for e in scores if not labels[e]]
    rng = random.Random(SEED); rng.shuffle(pos); rng.shuffle(neg); folds = [pos[i::5] + neg[i::5] for i in range(5)]
    tp = fp = 0
    for i in range(5):
        train = {e: scores[e] for j, f in enumerate(folds) if j != i for e in f}; t = fit(train, labels, max_fa)
        tp += sum(scores[e] >= t for e in folds[i] if labels[e]); fp += sum(scores[e] >= t for e in folds[i] if not labels[e])
    print(f'Cross-validated estimate: caught {tp}/{P} ({tp / P:.0%}), false flags {fp / N:.0%}. '
          f"Study result with {cfg['locked_study_result']['model']}: {cfg['locked_study_result']['caught']} at {cfg['locked_study_result']['false_flags']}.")
    if c / P < cfg['target']['aim_caught'] - 0.10: print('Warning: well below the 70% aim; review before running the full set.')
    CALIBRATION.write_text(json.dumps({'model': cfg['model'], 'cut_off': cut, 'caught': f'{c}/{P}', 'false_flags': f'{fa}/{N}',
                                       'cross_validated': f'{tp}/{P} caught, {fp / N:.1%} false flags', 'max_false_flags': max_fa,
                                       'time': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=2))
    print(f'Saved {CALIBRATION.relative_to(ROOT)}. Next: python run.py screen')

def cut_off(cfg, a):
    if a.cut_off is not None: return a.cut_off
    if not CALIBRATION.exists(): raise SystemExit('Run python run.py calibrate first (or pass --cut-off).')
    cal = json.loads(CALIBRATION.read_text())
    if cal['model'] != cfg['model']: raise SystemExit(f"Calibration is for {cal['model']}, config uses {cfg['model']}; recalibrate.")
    return cal['cut_off']

def cmd_screen(cfg, prompts, a):
    cut = cut_off(cfg, a); eps = load_episodes()
    ids = sorted(eps, key=lambda e: (eps[e]['source'] != 'rated', e))[:a.limit] if a.limit else sorted(eps)
    run_jobs(cfg, prompts, eps, ids, a.max_usd); cmd_report(cfg, prompts, a, cut)

def cmd_report(cfg, prompts, a, cut=None):
    cut = cut if cut is not None else cut_off(cfg, a); eps = load_episodes(); done, spent = load_results(cfg)
    rows = []
    for e, r in done.items():
        j = r['judgments']; x = eps.get(e, {})
        rows.append({'episode_id': e, 'source': r['source'], 'module': x.get('module', ''), 'language': x.get('language', ''),
                     'flagged': 'yes' if j['probability'] >= cut else 'no', 'probability': j['probability'], 'model_says_critical': j['critical'],
                     'types': ', '.join(j['types']), 'evidence_turns': ', '.join(j['evidence_turns']), 'reason': j['reason'],
                     'rater_critical': {True: 'yes', False: 'no'}.get(x.get('rater_critical'), 'not rated')})
    rows.sort(key=lambda r: (r['flagged'] != 'yes', -r['probability'], r['episode_id']))
    out = ROOT / 'outputs/critical-screen-flags.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{cfg['model']}, probability >= {cut}: scored {len(rows)}/{len(eps)} episodes, spent ${spent:.2f}")
    for src in ('rated', 'pool'):
        R = [r for r in rows if r['source'] == src]
        if R: print(f"  {src}: flagged {sum(r['flagged'] == 'yes' for r in R)}/{len(R)} ({sum(r['flagged'] == 'yes' for r in R) / len(R):.0%})")
    by = Counter((r['module'], r['flagged']) for r in rows)
    for m in sorted({r['module'] for r in rows}):
        n = by[(m, 'yes')] + by[(m, 'no')]; print(f'    module {m}: flagged {by[(m, "yes")]}/{n}')
    print(f'Review list: {out.relative_to(ROOT)} (flagged first). Keep outputs/ out of git.')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['status', 'test-call', 'calibrate', 'screen', 'report'])
    ap.add_argument('--max-usd', type=float, default=None, help='stop starting new calls at this total spend (default 15 calibrate, 150 screen)')
    ap.add_argument('--limit', type=int, help='screen: score only the first N episodes (rated first)')
    ap.add_argument('--cut-off', type=int, help='use this cut-off instead of outputs/calibration.json')
    a = ap.parse_args(); cfg, prompts = load_config()
    if a.max_usd is None: a.max_usd = 15.0 if a.command == 'calibrate' else 150.0
    {'status': cmd_status, 'test-call': cmd_test_call, 'calibrate': cmd_calibrate, 'screen': cmd_screen, 'report': cmd_report}[a.command](cfg, prompts, a)

if __name__ == '__main__':
    main()
