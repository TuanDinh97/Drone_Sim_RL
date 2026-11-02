# Contributing

quadsim is a research framework: contributions are welcome, and results must be reproducible and honestly reported.

## Setup and checks
- Environment: `bash scripts/setup_env.sh` (conda env `quadsim`; always install with `-c env/constraints.txt`).
- Before every pull request: `python -m pytest tests/core -q` (CPU, ~15 s). For changes to the simulation core, the
  tasks or the evaluator also run `bash scripts/validation/regression_evals.sh logs/eval/regress_<tag>` and
  `python scripts/validation/compare_evals.py logs/eval/regress_review logs/eval/regress_<tag>`: every changed metric
  must be explained by the change. Run `python scripts/validation/examples_check.py` when touching extension points.
- New behaviour comes with a unit test that fails without it (CPU-testable code must not import Isaac Lab).

## Adding components
Vehicles, sensors, action interfaces, tasks, terms, algorithms, baselines and benchmark suites are registered by
name; see `docs/FRAMEWORK.md` (architecture) and `docs/TUTORIALS.md` (recipes, `examples/`). Policy observations may
only contain deployable quantities; privileged information belongs to the critic group.

## Reporting results
- Evaluate with `scripts/rl/evaluate.py` or a benchmark suite (`scripts/benchmark/run.py`): fixed seeds, complete
  episodes, every environment contributing the same number of episodes, in- and out-of-distribution conditions.
- Train at least 3 seeds with identical configurations and pin `--num_envs`; report per-seed values and intervals
  (`quadsim.benchmark.stats`: stratified bootstrap CIs, IQM, probability of improvement), see `docs/BENCHMARK.md`.
- Report negative results and failed configurations next to the successful ones; never tune on the OOD conditions.
- Keep the provenance: run directories store configuration, command line and commit; curated results go to
  `docs/results/` through the collector scripts.

## Commits
Small, focused commits with a message that says what changed and why. Never commit logs, checkpoints, datasets or
`third_party/`; never force-push.
