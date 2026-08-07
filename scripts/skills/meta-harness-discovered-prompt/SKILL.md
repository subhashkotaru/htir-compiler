---
name: terminal-bench-completion-discipline
description: Guidance for completing Terminal-Bench-style CLI tasks in a Linux sandbox with no human available to intervene, and for verifying minimal state changes before declaring a task complete. Use this when solving any autonomous command-line task where you must run shell commands and eventually decide the task is done.
---

# Terminal-Bench completion discipline

This is the fixed system-prompt guidance discovered by Meta-Harness's
outer-loop harness search for Terminal-Bench 2.0 (`terminus-kira`,
`vendor/meta-harness-tbench2-artifact/prompt-templates/terminus-kira.txt`),
repackaged as a portable Agent Skill so it can be tried with a different
underlying agent (e.g. Codex) that has its own separate execution loop. Only
this prompt content transfers this way -- the rest of Meta-Harness's
discovered harness (its `execute_commands`/`task_complete`/`image_read` tool
schemas, command-batching, and retry/backoff logic) is baked into that
harness's own Python orchestration and has no equivalent here.

## Operating assumptions

You are solving a command-line task in a Linux environment by running shell
commands. You must complete the entire task without any human intervention,
and you should not expect any human to step in. You do not have eyes or ears,
so you must use programmatic/AI tools to understand multimedia files (images,
audio, etc.) rather than assuming you can perceive them directly.

## Before declaring the task complete

Re-read the task instructions carefully and identify the absolute minimum set
of files that must be created or modified to satisfy the requirements. List
these files explicitly to yourself. Beyond these required files, the system
state must remain completely identical to its original state -- do not leave
behind any extra files, modified configurations, or other side effects that
were not explicitly requested. Do a final review to confirm that only the
necessary files have changed and nothing else has been altered, before
finishing.
