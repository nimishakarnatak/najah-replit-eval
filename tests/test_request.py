"""Synthetic-episode checks that najah_eval sends the same request and applies the same validation as the study run
(no conversation data, no API calls). Needs the study project next to this repo; skipped when it is absent (Replit).

Usage: python tests/test_request.py [--project ~/Desktop/najah/najah-llm-judge]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from najah_eval.screen import load_config, build_request, validate

EPISODE = {'episode_id': 'TEST-1', 'module': 'cv_builder', 'language': 'ar-fr', 'delivery_evidence': [],
           'turns': [{'turn_id': '001', 'role': 'assistant', 'segment': 'prior', 'text': 'Bonjour !', 'text_en': 'Hello!'},
                     {'turn_id': '002', 'role': 'user', 'segment': 'episode', 'text': 'بغيت نقاد CV', 'text_en': 'I want to make a CV'},
                     {'turn_id': '003', 'role': 'assistant', 'segment': 'episode', 'text': "D'accord, commençons.", 'text_en': "OK, let's start."}]}
ANSWERS = [{'critical': 'no', 'probability': 6, 'types': [], 'evidence_turns': [], 'reason': 'ok'},
           {'critical': 'yes', 'probability': '55', 'types': ['fabrication'], 'evidence_turns': ['3'], 'reason': 'x'},
           {'critical': 'yes', 'probability': 40, 'types': ['fabrication'], 'evidence_turns': [], 'reason': 'x'},
           {'critical': 'maybe', 'probability': 40}, {'critical': 'no', 'probability': 140}, {'critical': 'yes', 'probability': 50, 'types': ['other'], 'evidence_turns': ['002']}]

def main():
    proj = Path(sys.argv[sys.argv.index('--project') + 1] if '--project' in sys.argv else Path.home() / 'Desktop/najah/najah-llm-judge').expanduser()
    if not (proj / 'najah/judges.py').exists(): print('study project not found; skipped'); return
    sys.path[:0] = [str(proj), str(proj / 'scripts')]
    from najah.judges import SYSTEM
    from run_extra import input_data
    from run_critical_variants import prompt_text, FORMAT, validate as study_validate
    cfg, prompts = load_config(); req = build_request(cfg, prompts, EPISODE)
    assert req['system'] == SYSTEM, 'system prompt differs'
    assert req['messages'][0]['content'][0]['text'] == json.dumps({'rubric': prompt_text('B'), 'episode': input_data(EPISODE)}, ensure_ascii=False), 'user text differs'
    assert req['messages'][0]['content'][1]['text'] == FORMAT, 'answer format differs'
    assert 'text_en' not in req['messages'][0]['content'][0]['text'], 'English translation leaked into the request'
    for ans in ANSWERS:
        def run(fn, *args):
            try: return fn(*args)
            except ValueError as exc: return f'error: {exc}'
        assert run(validate, ans, EPISODE) == run(study_validate, ans, EPISODE, 'B'), f'validation differs for {ans}'
    print(f'OK: system prompt, user text, answer format and validation ({len(ANSWERS)} answers) match the study run; model {cfg["model"]}.')

if __name__ == '__main__':
    main()
