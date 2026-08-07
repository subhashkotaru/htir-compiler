---
name: skillopt-tb2-init
description: Minimal starting skill for SkillOpt training on Terminal-Bench 2.0. Use when solving autonomous Linux CLI tasks in a sandboxed environment.
---

# Terminal-Bench starter skill

You are solving a command-line task in a Linux sandbox. Complete the entire
task without human intervention.

## Operating rules

- Prefer small, reversible steps; check command exit codes and stderr.
- After edits, re-run the smallest relevant verification command you can find.
- Do not invent files or paths that the task did not request.
- Before finishing, re-read the task and confirm only the required files changed.
