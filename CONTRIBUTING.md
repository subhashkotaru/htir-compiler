# Contributing to HTIR

Thanks for your interest! HTIR aims to be a broadly useful, framework-neutral
verification layer for agent trajectories, and it is designed to be extended
**without forking core**. The three highest-leverage contributions are:

1. **Trace adapters** — support a new agent framework (`htir/adapters/`).
2. **Domain specs** — a new environment's operations, artifacts, constraints,
   and obligation templates (`htir/domains/*.yaml`).
3. **Checkers** — new mechanical checks for a claim type or template
   (`htir.agents.checker_registry`).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The deterministic pipeline runs with **no API key**. Only tests that exercise
LLM-backed passes need one, and they stub the client — so `pytest` is fully
offline and reproducible. Please keep it that way: gate any new LLM call behind
a flag that defaults to off.

## Adding a trace adapter

1. Create `htir/adapters/<framework>.py` with a `TraceAdapter` subclass; set a
   unique `name`, implement `detect` and `parse`, and decorate with
   `@register_adapter`.
2. `parse` returns canonical step dicts — use `canonical_step(...)` and
   `tool_call(...)` from `htir.adapters.base`. Preserve **structured** tool
   calls; never flatten them into the response string.
3. Make `detect` specific so autodetection doesn't collide with other formats.
4. Import your module in `htir/adapters/__init__.py` (built-in) or ship it as a
   `htir.adapters` entry point (third-party package).
5. Add a round-trip test in `tests/`.

## Adding a checker

Register a `(CheckerContext) -> CheckerResult` callable keyed by `claim_type`,
`template_id`, or `required_evidence`. Checkers must only inspect the
obligation's **local** neighbourhood and **abstain** rather than guess when
evidence is missing.

## Ground rules

- **Determinism:** the non-LLM path must stay byte-for-byte reproducible.
- **Backward compatibility:** don't break the serialized HTIR schema without
  bumping `HTIR_SCHEMA_VERSION`.
- **Tests:** add coverage for new behavior; keep `pytest` green.
- **Style:** match the surrounding code; type hints on public functions.

## Pull requests

Keep PRs focused. Describe what changed and why, link any issue, and note
whether the HTIR schema or any fixture changed. By contributing you agree your
contributions are licensed under Apache-2.0.
