"""Variant-B critical-failure screen: request building, provider call and answer validation.

Mirrors the calibrated study run in najah-llm-judge (scripts/run_critical_variants.py calling najah/judges.py _post):
same system prompt, same user content (JSON of rubric + episode, then the answer format), same validation rules.
Two deliberate differences: no prompt-cache marker (each prompt is used once, and a cache write costs extra), and the
"judgment shape" unwrap seen with larger Claude models (each field wrapped as {"value", "evidence_turns", "reason"}).
Labels, participant codes and other metadata never enter a request: only module, language, turns and delivery evidence.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ['fabrication', 'unsafeAdvice', 'privacyViolation', 'stereotypingDiscrimination',
          'missingEscalation', 'manipulativeAuthority', 'otherSeriousFailure']

def load_config():
    cfg = json.loads((ROOT / 'config/screen.json').read_text(encoding='utf-8'))
    prompts = {k: (ROOT / p).read_text(encoding='utf-8') for k, p in cfg['prompt_files'].items()}
    if hashlib.sha256(prompts['rubric'].encode()).hexdigest()[:16] != cfg['rubric_sha256_16']:
        raise SystemExit('prompts/critical_variant_B.txt differs from the locked screen prompt; restore it before running.')
    return cfg, prompts

def input_data(episode):
    turns = [{k: v for k, v in t.items() if k != 'text_en'} for t in episode['turns']]
    return {'module': episode['module'], 'language': episode['language'], 'turns': turns, 'delivery_evidence': episode.get('delivery_evidence', [])}

def build_request(cfg, prompts, episode):
    shared = json.dumps({'rubric': prompts['rubric'], 'episode': input_data(episode)}, ensure_ascii=False)
    return {'model': cfg['model'], 'system': prompts['system'], 'max_tokens': cfg['max_tokens'],
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': shared}, {'type': 'text', 'text': prompts['format']}]}]}

def client():
    """Replit-managed credentials when the Anthropic AI Integration is enabled; otherwise ANTHROPIC_API_KEY (own key)."""
    import anthropic
    base, key = os.environ.get('AI_INTEGRATIONS_ANTHROPIC_BASE_URL'), os.environ.get('AI_INTEGRATIONS_ANTHROPIC_API_KEY')
    if base and key:
        return anthropic.Anthropic(api_key=key, base_url=base, max_retries=4, timeout=300), 'Replit-managed'
    if os.environ.get('ANTHROPIC_API_KEY'):
        return anthropic.Anthropic(max_retries=4, timeout=300), 'own key (ANTHROPIC_API_KEY)'
    raise SystemExit('No credentials: enable the Anthropic (Replit managed) AI Integration, or add ANTHROPIC_API_KEY in Replit Secrets.')

def normalize_turn(x, turn_ids):
    """Map an unpadded numeric ID ('2') to the episode's padded form ('002'); other IDs pass through unchanged."""
    x = str(x).strip()
    if x in turn_ids or not x.isdigit():
        return x
    padded = {t.lstrip('0') or '0': t for t in turn_ids if t.isdigit()}
    return padded.get(x.lstrip('0') or '0', x)

def unwrap(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get('critical'), dict): return obj, False
    c = obj['critical']; val = lambda v: v.get('value') if isinstance(v, dict) else v
    types = val(obj.get('types'))
    flat = {'critical': c.get('value'), 'probability': val(obj.get('probability')),
            'types': types if isinstance(types, list) else [t.strip() for t in str(types or '').split(',') if t.strip() and t.strip().lower() != 'none'],
            'evidence_turns': c.get('evidence_turns') or [], 'reason': c.get('reason', '')}
    return flat, True

def validate(obj, episode):
    turn_ids = {str(t['turn_id']) for t in episode['turns']}
    if not isinstance(obj, dict): raise ValueError('object required')
    crit = str(obj.get('critical', '')).lower()
    if crit not in ('yes', 'no'): raise ValueError('critical must be yes or no')
    try: prob = int(round(float(obj.get('probability'))))
    except (TypeError, ValueError): raise ValueError('probability must be a number')
    if not 0 <= prob <= 100: raise ValueError('probability must be 0-100')
    types = [str(t) for t in (obj.get('types') or [])]
    if any(t not in BINARY for t in types): raise ValueError(f'unknown types {types[:3]}')
    ev = [normalize_turn(x, turn_ids) for x in (obj.get('evidence_turns') or [])]
    if any(x not in turn_ids for x in ev): raise ValueError('evidence references a nonexistent turn')
    if crit == 'yes' and not ev: raise ValueError('critical yes requires evidence turns')
    return {'critical': crit, 'probability': prob, 'types': types if crit == 'yes' else [], 'evidence_turns': ev, 'reason': str(obj.get('reason', ''))[:800]}

def parse(text):
    if text.strip().startswith('```'): text = '\n'.join(text.strip().splitlines()[1:-1])
    return json.loads(text)

def score(cl, cfg, prompts, episode, attempts=2):
    """One episode; one retry for an answer-format failure (HTTP errors are retried by the SDK)."""
    usage = {'input_tokens': 0, 'output_tokens': 0}; err = None
    for _ in range(attempts):
        msg = cl.messages.create(**build_request(cfg, prompts, episode))
        usage['input_tokens'] += msg.usage.input_tokens; usage['output_tokens'] += msg.usage.output_tokens
        try:
            if msg.stop_reason != 'end_turn': raise ValueError(f'stop_reason {msg.stop_reason}; not scored')
            obj, wrapped = unwrap(parse(''.join(b.text for b in msg.content if b.type == 'text')))
            return {'judgments': validate(obj, episode), 'unwrapped_judgment_shape': wrapped, 'returned_model': msg.model,
                    'response_id': msg.id, 'usage': usage}
        except (ValueError, json.JSONDecodeError) as exc:
            err = str(exc)[:300]
    raise ValueError(err, usage)
