# Getting Started

## 1) Install and verify environment

```bash
# from repository root
python -m unittest discover -s tests -v
python completeness/runner.py --mode fixture --fail-under 1.0
```

## 2) Configure API keys

Set one or more:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## 3) First request

```python
from lm15 import Message, LMRequest, Part, build_default

lm = build_default(use_pycurl=True)
req = LMRequest(
    model="gpt-4.1-mini",
    messages=(Message(role="user", parts=(Part.text_part("Reply with exactly: ok"),)),),
)
resp = lm.complete(req)
print(resp.message.parts[0].text)
```

## 4) Stream request

```python
for event in lm.stream(req):
    if event.type == "delta" and event.delta and event.delta.get("type") == "text":
        print(event.delta["text"], end="")
```

## 5) Add middleware

```python
from lm15 import with_cache, with_history, with_retries

history = []
cache = {}

lm.middleware.complete_mw.append(with_cache(cache))
lm.middleware.complete_mw.append(with_history(history))
lm.middleware.complete_mw.append(with_retries(max_retries=2))
```

## 6) Run cookbook examples

```bash
python examples/01_basic_text.py
python examples/02_streaming.py
python examples/03_tools.py
```
