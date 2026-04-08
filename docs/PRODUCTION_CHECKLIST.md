# Production Checklist

## Release gates

- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] `python completeness/runner.py --mode fixture --fail-under 1.0` passes.
- [ ] `python completeness/runner.py --mode all --fail-under 1.0` passes.
- [ ] Cookbook example scripts execute without import/runtime errors.
- [ ] Import benchmark regression checked (`bench_import.py`).

## Reliability gates

- [ ] Retry policy configured.
- [ ] Timeout policy configured.
- [ ] Cache policy configured.
- [ ] Error taxonomy mapped and monitored.

## Provider gates

- [ ] `supports` flags match implementation.
- [ ] `manifest` auth/variants documented.
- [ ] Complete + stream normalized.
- [ ] Extended endpoints tested where marked supported.

## Observability gates

- [ ] History middleware enabled in staging.
- [ ] Provider payload passthrough sampled for drift detection.
- [ ] Completeness report archived (`completeness/history`).

## Security gates

- [ ] API keys from env/secret manager only.
- [ ] No key logging.
- [ ] External plugin sources pinned/verified.
