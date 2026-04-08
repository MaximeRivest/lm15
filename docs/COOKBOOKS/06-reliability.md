# Cookbook 06 — Reliability (Retries, Cache, History, Timeouts)

## Goal

Harden runtime behavior without changing adapters.

## Example

```python
from lm15 import Message, LMRequest, Part, TransportPolicy, build_default, with_cache, with_history, with_retries

policy = TransportPolicy(timeout=30.0, connect_timeout=10.0, read_timeout=30.0, max_retries=1)
lm = build_default(use_pycurl=True, policy=policy)

history = []
cache = {}
lm.middleware.complete_mw.append(with_cache(cache))
lm.middleware.complete_mw.append(with_history(history))
lm.middleware.complete_mw.append(with_retries(max_retries=2, sleep_base=0.2))

req = LMRequest(model="gpt-4.1-mini", messages=(Message(role="user", parts=(Part.text_part("Reply with ok"),)),))
print(lm.complete(req).message.parts[0].text)
print("history events:", len(history))
```

## Related runnable script

- `examples/06_reliability.py`
