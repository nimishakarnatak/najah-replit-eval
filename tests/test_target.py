"""Offline check of the team target in run.py (no API calls, no conversation data): calibrate picks the cut-off with the
most critical cases caught within 20% false flags, and screen refuses to run when that catches fewer than 70%.

Usage: python tests/test_target.py
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run

def scenario(tmp, crit_scores, other_scores):
    eps = {}; results = []
    for i, (crit, s) in enumerate([(True, x) for x in crit_scores] + [(False, x) for x in other_scores]):
        e = f'R{i:03d}'; eps[e] = {'episode_id': e, 'source': 'rated', 'rater_critical': crit, 'turns': []}
        results.append({'episode_id': e, 'source': 'rated', 'model': cfg['model'], 'status': 'ok', 'usage': {},
                        'judgments': {'probability': s, 'critical': 'no', 'types': [], 'evidence_turns': [], 'reason': ''}})
    run.RESULTS = tmp / 'results.jsonl'; run.CALIBRATION = tmp / 'calibration.json'
    run.RESULTS.write_text(''.join(json.dumps(r) + '\n' for r in results))
    run.load_episodes = lambda: eps; run.run_jobs = lambda *a, **k: None
    run.cmd_calibrate(cfg, prompts, SimpleNamespace(max_usd=0))
    return json.loads(run.CALIBRATION.read_text())

cfg, prompts = run.load_config()
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    # 10 critical, 50 other. Meets: cut-off 20 catches 8/10 with 10/50 = 20% false flags.
    cal = scenario(tmp, [90, 80, 60, 50, 40, 30, 25, 20, 5, 3], [30] * 5 + [20] * 5 + [10] * 20 + [5] * 20)
    assert cal['cut_off'] == 20 and cal['caught'] == '8/10' and cal['meets_target'], cal
    assert run.cut_off(cfg, SimpleNamespace(cut_off=None, accept_below_target=False)) == 20
    # Below target: within 20% false flags only 5/10 are caught, so screen must refuse unless overridden.
    cal = scenario(tmp, [90, 80, 60, 50, 40, 8, 7, 6, 5, 3], [30] * 5 + [20] * 5 + [10] * 20 + [5] * 20)
    assert cal['caught'] == '5/10' and not cal['meets_target'], cal
    try:
        run.cut_off(cfg, SimpleNamespace(cut_off=None, accept_below_target=False)); raise AssertionError('screen should have refused')
    except SystemExit as exc:
        assert 'below the 70% target' in str(exc), exc
    assert run.cut_off(cfg, SimpleNamespace(cut_off=None, accept_below_target=True)) == cal['cut_off']
print('OK: calibrate fits within 20% false flags; screen runs at >= 70% caught and refuses below it unless overridden.')
