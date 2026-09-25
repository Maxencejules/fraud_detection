# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) 0.12 or newer and Docker with Compose v2.
uv installs the pinned Python version and every dependency from `uv.lock`.

```bash
uv sync --all-extras          # create .venv with all services' dependencies
uvx pre-commit install        # run lint and format checks on every commit
```

Without `make` (e.g. on Windows), run the commands behind each target in the
[Makefile](Makefile) directly.

## Workflow

1. Create a branch from `main`.
2. Make the change with tests. Behaviour changes need a test that fails without them.
3. Run `make check` (ruff, mypy `--strict`, unit tests with the 85% coverage gate).
4. For changes that touch services, Docker or Compose, also run the stack:
   `make up && make smoke && make test-integration`.
5. Open a pull request. CI runs the same checks, plus the end-to-end job, which builds
   every image, starts the stack, runs the smoke and integration tests, and records a
   benchmark.

## Conventions

- **Dependencies.** Add runtime dependencies to the optional group of the service
  that needs them (`stream`, `serving`, `training`, `monitoring`), never to all of
  them. `tests/unit/test_dependency_boundaries.py` imports every service with only
  its own group installed and fails if a module reaches outside it. Run `uv lock`
  after editing `pyproject.toml`.
- **Features.** The model contract is the field order of `FeatureVector` in
  `schemas.py`. Adding or changing a feature means updating `features.py`, its tests
  and the feature table in `docs/architecture.md`, then re-running bootstrap and
  training. The predictor refuses models trained on a different contract.
- **Configuration.** Every setting is a typed field in `config.py`, read from the
  environment, with a safe default. Document new variables in `.env.example`.
- **Logging.** Use the standard `logging` module with a short event name and
  structured `extra` fields: `logger.info("model_loaded", extra={"version": v})`.
  Output is one JSON object per line.
- **Metrics.** Name new Prometheus metrics `fraud_<service>_<what>_<unit>` and add a
  panel or alert when the metric matters operationally.
- **Results.** Quote only measured numbers, together with how and where they were
  measured (see `docs/results/`).
- **Changelog.** Record user-visible changes in the topmost, unreleased section of
  [`CHANGELOG.md`](CHANGELOG.md), and put anything that breaks existing users under
  "Breaking changes".

## Commit messages

Use the imperative mood ("Add scorer retry backoff"), keep the subject under about
72 characters, and explain *why* in the body when it is not obvious.
