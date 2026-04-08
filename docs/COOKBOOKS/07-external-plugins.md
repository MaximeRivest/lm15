# Cookbook 07 — External Provider Plugins (`lm15-x-*`)

## Goal

Ship providers without core PRs via entry points.

## External package `pyproject.toml`

```toml
[project]
name = "lm15-x-myprovider"
version = "0.1.0"
dependencies = ["lm15"]

[project.entry-points."lm15.providers"]
myprovider = "lm15_x_myprovider:build_adapter"
```

## External package module

```python
# lm15_x_myprovider.py
from lm15.features import EndpointSupport, ProviderManifest

class MyAdapter:
    provider = "myprovider"
    capabilities = None
    supports = EndpointSupport(complete=True, stream=True)
    manifest = ProviderManifest(provider="myprovider", supports=supports)

    def complete(self, request):
        ...

    def stream(self, request):
        ...


def build_adapter():
    return MyAdapter()
```

## Load automatically

```python
from lm15 import build_default
lm = build_default(discover_plugins=True)
```

## Load explicitly

```python
from lm15 import UniversalLM, load_plugins

lm = UniversalLM()
result = load_plugins(lm)
print(result.loaded, result.failed)
```

## Related runnable script

- `examples/07_external_plugins.py`
