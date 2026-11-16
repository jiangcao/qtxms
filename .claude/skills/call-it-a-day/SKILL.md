---
name: call-it-a-day
description: End-of-day wrap-up for this project. Use when the user says "call it a day", "wrap up", "save the session", or asks to record today's work. Checks running jobs, saves the conversation in sessions/, writes an explainer for others in reports/, records findings and next steps in notes/, and updates CLAUDE.md and job.sh.
---

# Call it a day

Wrap up the working day so that tomorrow's session, and a reader who was not here, can pick it up. Do every step;
do not ask for confirmation between them. Use today's date (YYYY-MM-DD) and a short topic slug, e.g.
`2026-11-15-density-matrix-hierarchy`. Use the same slug in all three places.

## 1. Check what is running and collect the last results

- List the user's running jobs (python drivers, run scripts, queue waiters) and the GPU state (`nvidia-smi`).
- Read the tail of every log touched today. Runs that finished since the last report: get their numbers. Runs that
  failed: get the error and how far they got. Runs still going: say so, with the log path and what they will produce.
- Never wait for a job to finish. Do not stop a job unless the user asked.

## 2. Session log: `sessions/<date>-<slug>.txt`

Plain text, the format of the existing files in `sessions/` (title line, `====` underline, numbered sections with
`----` underlines, lines up to ~120 characters). Chronological record of the conversation:

- what the user asked or proposed, in their terms, and the decisions they made (with the reason);
- the discussion: objections raised, which were conceded, which were confirmed by a measurement;
- every run: script, log file, host, wall time, and the numbers (tables where there are several cases);
- code written or changed: file, what it does, the options added;
- negative results and failed runs, with the cause if known;
- mistakes made on the way and their fixes (wrong kill, wrong default, wrong number reported);
- housekeeping: caches created and where, files moved, what was added to job.sh / CLAUDE.md.

## 3. Explainer for others: `reports/<date>-<slug>/summary.md`

For a reader who knows the field but was not in the session (the reports go to collaborators). Follow the existing
reports (`reports/*/summary.md`):

- open with what the method is and the main result, including whether it is actually better than the baseline;
- explain the method from the ground up: the problem, each building block, each term used in the result tables
  (define every label that appears in a table or a log line);
- small text diagrams for structure (partitions, levels, cycles) when they help;
- results as tables with the device, energy, tolerance and what the numbers are (e.g. iterations min-max over columns);
- limits and open points, stated as plainly as the successes;
- how to run it (the actual commands, pointing to job.sh).

Add figures with a `make_figures.py` next to the summary only when a plot shows something a table does not. Use the
project naming from CLAUDE.md (slab / tile, level 1 / level 2, "local rho" / "global rho").

## 4. Findings, lessons, next steps: `notes/<date>-<slug>.md`

Three sections, numbered items, each with the numbers that support it:

- **Findings**: what was measured today, positive and negative, one item per result.
- **What we learnt**: the conclusions that should change how we work next (what makes it work, what the failure
  mechanism is, what is still not known). Mark anything that is a reading and not a verified cause.
- **Next steps**: ordered by priority, concrete enough to start from (which script, which device, which option),
  including reruns of failed cases and caches that already exist.

Start the file with pointers to the report, the session log and the code.

## 5. Update the project records

- `notes/DECISIONS.md` (the full "Decisions and why" log; NOT `CLAUDE.md`, which keeps only a short digest that
  changes when a standing decision changes): add today's results to the relevant entry in the existing dense style (script and
  log names, numbers, the conclusion after "=>", open items, pointers to the three documents). Correct statements
  that today's results made wrong. Do not duplicate what is already there.
- `job.sh`: every run script started today must be listed; mark each line with its status and outcome
  (`# done (164 s): ...`, `# stopped ...`, `# failed: ...`, `# running since ...`). Finished or queued-by-waiter lines
  stay commented out so that running job.sh does not start them twice.
- Memory: save only durable facts that are not already in CLAUDE.md (a new host constraint, a user preference).

## 6. Report back

A short final message: what is still running (or that nothing is), any run that failed and what that means, the
paths of the three documents, and the top two or three next steps. No recap of the whole day.

## Rules

- Big files and caches go to `/scratch/jiacao`, never to the repo disk (check `df` before writing anything large).
- Report numbers exactly as in the logs; if a run has no result, say so and do not fill the gap.
- Do not commit or push unless the user asks.
