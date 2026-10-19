---
name: sdd
description: Spec-driven development workflow for this repo — start a new effort in efforts/
  (spec → tasks → implement → verify → close-out with plain-language outcome). Use when the
  user wants to start a new piece of work that will take more than ~half a day, or when
  asked to "start/log/close an effort".
---

# SDD — efforts lifecycle in this repo

Work is organized by **effort**: `efforts/YYYY-MM-DD-<slug>/` with `spec.md`
(goal/expected/non-goals/acceptance criteria), `tasks.md` (dated checkboxes), `outcome.md`
(plain-language close-out). Index: `efforts/README.md`. Now: `STATE.md`. Not-yet-done:
`BACKLOG.md`.

**Threshold rule:** only work expected to take >~½ day, touch multiple files, or change an
interface gets the full ceremony. Smaller tasks: just do them, log a line in the owning
effort's tasks.md (or BACKLOG).

## 1. Start an effort
1. `mkdir efforts/$(date +%F)-<short-slug>/`, copy `efforts/_TEMPLATE/{spec,tasks,outcome}.md`.
2. Fill spec.md: Goal · Expected results · **Non-goals** · Acceptance criteria as verify
   commands. This repo verifies via:
   - `../function_calling_edge_fine_tuning/train/venv/bin/python eval/score_no_lang.py --dump eval/results/<tag>_preds.jsonl [--canon]` (official benchmark-v2 F1/EM)
   - `../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_extract.py --candidate <c> [--adapter train/runs/<c>/best | --model-dir <merged>] --gpu 1`
   - dataset checks: row counts in `data/*_stats.json` / `data/v3_stats.json`
   `Status: draft`, `Start:` today.
3. **Spec review gate:** show the spec to the user before implementing.
4. Fill tasks.md plan + first checkboxes. Pre-register acceptance rules for model runs
   (this repo's tradition — see round 8/9/12 tasks.md).

## 2. Implement
- Delegate coding to the `python-coder` subagent with acceptance criteria in the prompt;
  require the verification command in its report (per AGENTS.md workflow).
- Check off tasks with dates.
- GPU: pin `--gpu 1`; sibling fcft jobs may occupy GPUs (see AGENTS.md Gotchas).

## 3. Verify
Run every acceptance command yourself. Failing → not done. Append results to
`reports/RESULTS.md` when a round closes (canonical tables live there).

## 4. Close out (while context is still loaded)
1. outcome.md: plain language — did / worked / didn't / bottom line (plain-results-recap
   style; concrete numbers, analogies welcome).
2. spec.md: Status + End date.
3. tasks.md: final status header; unfinished → BACKLOG.md.
4. efforts/README.md: index row.
5. STATE.md: refresh live numbers from `reports/RESULTS.md` + `data/v3_stats.json`.
6. Durable lesson → AGENTS.md conventions (Layout/Gotchas).
