# Najah critical-failure screen — Replit run

Runs the team's locked critical-failure screen on the full Najah conversation set (300 rated + 3,263 unrated module
episodes) using Replit-managed Claude, and writes a review list of flagged episodes.

**The screen** (team decision, 19 Oct 2026): prompt variant B (the instrument's critical-failure question and
definitions, base rate, and clarifications), one call per episode, flag when the model's probability ≥ cut-off.
In the study, with Claude Sonnet 5.5 and cut-off 12, it caught 19/26 critical cases (73%) at 19% false flags;
cross-validated 65% at 18%.

**Target (enforced in code).** At least **70%** of the rated critical cases caught with at most **20%** false flags
(`config/screen.json` → `target`). `calibrate` picks the cut-off that catches the most rated critical cases within 20%
false flags and records whether it reaches 70%; `screen` and `report` refuse to run when it does not, unless you pass
`--accept-below-target`. `report` also prints the catch and false-flag rates on the rated episodes, and warns if the
pool is flagged at a very different rate. Checked by `tests/test_target.py`.

**Model note.** Replit-managed AI Integrations do not offer `claude-sonnet-5-5` (Replit docs, 2 Oct 2026), so this
repo uses `claude-sonnet-5` (same price). Because the probability scale can differ between models, `calibrate` first
scores the 300 rated episodes and refits the cut-off to the target above. If `claude-sonnet-5` cannot reach 70% within
20% false flags, use your own key with `claude-sonnet-5-5` (see the last section), which met it in the study.

## Files

| Path | What it is |
|---|---|
| `config/screen.json` | Model, token limit, target, prices, locked study result |
| `prompts/` | System prompt, variant-B prompt and answer format, copied verbatim from the study (hash-checked) |
| `najah_eval/screen.py` | Request building, Claude call, answer validation (same rules as the study) |
| `najah_eval/data.py` | Loads the conversation parts (encrypted or plain) |
| `run.py` | `status`, `test-call`, `calibrate`, `screen`, `report` |
| `prepare_data.py` | Builds the data parts on the study computer (not used on Replit) |
| `tests/test_request.py` | Checks the requests and validation match the study run (synthetic episode, no API calls) |
| `tests/test_target.py` | Checks the 70% / 20% target gate (synthetic scores, no API calls) |
| `data/` | Encrypted conversation parts (`conversations-NNN.jsonl.gz.enc`), added by step 1 |
| `outputs/` | Results, calibration and the review list (git-ignored) |

## 1. Build the data (study computer, once)

```bash
cd ~/Desktop/najah/najah-replit-eval
.venv/bin/python prepare_data.py
```

This reads `../najah-llm-judge` (split-2 database and `data/challenge_pool_episodes.json`), keeps only what the judge
sees plus the episode id, source and the rated label (critical = either rater flagged), checks that every request
matches the study run, and writes encrypted parts to `data/`. The key goes to `~/Desktop/najah/najah-data-key.txt`,
outside the repo. Use `--plain` instead to write unencrypted parts for a direct upload to Replit (never to GitHub).

## 2. Push to a private GitHub repository

Create an empty **private** repository on GitHub (no README or licence), then:

```bash
cd ~/Desktop/najah/najah-replit-eval
git add data/*.jsonl.gz.enc && git commit -m "Add encrypted conversation data"
git remote add origin https://github.com/<you>/najah-replit-eval.git
git push -u origin main
```

## 3. Set up Replit

1. **Create Repl → Import from GitHub**, connect your GitHub account and pick the private repository.
2. Turn on Replit-managed Claude: ask Replit Agent *"Add the Anthropic (Replit managed) AI integration to this app. Do
   not change any code."* This needs a paid Replit plan; on Pro or Enterprise an organisation admin must enable it.
3. **Tools → Secrets**: add `NAJAH_DATA_KEY` with the contents of `~/Desktop/najah/najah-data-key.txt`.
4. In the Shell: `pip install -r requirements.txt`

## 4. Run (Replit Shell)

```bash
python run.py status          # data counts, credentials ("Replit-managed"), progress, spend
python run.py test-call       # one call, about $0.03
python run.py calibrate       # 300 rated episodes, about $9; refits the cut-off and checks the 70% / 20% target
python run.py screen          # the remaining 3,263 episodes, about $90-100; stops at --max-usd (default 150)
python run.py report          # outputs/critical-screen-flags.csv, flagged episodes first
```

Runs resume after an interruption; failed episodes are retried when you rerun the same command. Download
`outputs/critical-screen-flags.csv` and `outputs/calibration.json` when done.

**Cost.** About $0.03 per episode (about 11,000 input and 600 output tokens at $2/$10 per million), so about $100 for
all 3,563 episodes, billed to Replit credits at the public API price. `status` shows the running total.

## Privacy

- The conversations contain participants' personal details. Keep the GitHub repository and the Repl **private**, and
  check that your data agreement and ethics approval cover GitHub and Replit as processors.
- The unrated pool has automated de-identification only; its manual privacy review is pending.
- Data in git is encrypted; the key lives only in your key file and the Replit Secret. Replit decrypts it at run time.
- Requests carry only module, language, turns and delivery evidence; labels and participant codes are never sent.
- `outputs/` quotes conversations in the model's reasons; it is git-ignored, so don't force-add it.

## Using your own Anthropic key instead

To keep the study model, set `"model": "claude-sonnet-5-5"` in `config/screen.json`, add `ANTHROPIC_API_KEY` in
Secrets and do not enable the Replit integration (Replit-managed credentials take precedence when present). The study
cut-off then applies: `python run.py screen --cut-off 12` (or still run `calibrate` to check it).
