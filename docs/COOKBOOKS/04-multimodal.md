# Cookbook 04 — Multimodal Input

## Goal

Send text + image/document/audio with normalized `Part` and `DataSource`.

## Example (image URL)

```python
from lm15 import DataSource, LMRequest, Message, Part, build_default

lm = build_default(use_pycurl=True)

image = Part(
    type="image",
    source=DataSource(type="url", url="https://example.com/cat.jpg", media_type="image/jpeg", detail="low"),
)

req = LMRequest(
    model="gemini-2.0-flash-lite",
    messages=(
        Message(role="user", parts=(image, Part.text_part("Describe this image in one sentence."))),
    ),
)

resp = lm.complete(req, provider="gemini")
print(resp.message.parts[0].text)
```

## Related runnable script

- `examples/04_multimodal.py`
