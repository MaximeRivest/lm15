# Cookbook 08 — models.dev Hydration

## Goal

Hydrate capability resolver from `https://models.dev/api.json`.

## Example

```python
from lm15 import build_default, fetch_models_dev, hydrate_with_specs

specs = fetch_models_dev()
hydrate_with_specs(specs)

lm = build_default(hydrate_models_dev_catalog=False)
# resolver now uses hydrated model catalog
```

Or directly:

```python
lm = build_default(hydrate_models_dev_catalog=True)
```

## Notes

- Use models.dev for model discovery/capability hints.
- Do not treat it as wire-contract truth.

## Related runnable script

- `examples/08_models_dev_hydration.py`
