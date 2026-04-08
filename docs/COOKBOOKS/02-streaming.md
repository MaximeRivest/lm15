# Cookbook 02 — Streaming

## Goal

Consume normalized stream deltas from different providers using one handler.

## Example

```python
from lm15 import Message, LMRequest, Part, build_default

lm = build_default(use_pycurl=True)
req = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message(role="user", parts=(Part.text_part("Write 3 short bullet points about HTTP/2."),)),),
)

for event in lm.stream(req):
    if event.type == "start":
        print("[start]")
    elif event.type == "delta" and event.delta:
        if event.delta.get("type") == "text":
            print(event.delta.get("text", ""), end="")
    elif event.type == "end":
        print("\n[end]", event.usage)
    elif event.type == "error":
        print("\n[error]", event.error)
```

## Related runnable script

- `examples/02_streaming.py`
