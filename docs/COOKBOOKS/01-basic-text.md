# Cookbook 01 — Basic Text Calls

## Goal

Run one normalized request across providers.

## Example

```python
from lm15 import Message, LMRequest, Part, build_default

lm = build_default(use_pycurl=True)

req = LMRequest(
    model="claude-sonnet-4-5",
    messages=(Message(role="user", parts=(Part.text_part("Summarize TCP in one sentence."),)),),
)

resp = lm.complete(req)
print(resp.message.parts[0].text)
print(resp.usage)
```

## Force specific provider

```python
resp = lm.complete(req, provider="anthropic")
```

## Related runnable script

- `examples/01_basic_text.py`
