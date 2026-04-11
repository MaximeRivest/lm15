# Cookbook 03 — Tool Calling

## Goal

Define function tools, detect tool calls, execute tool, return tool result.

## Example

```python
from lm15 import FunctionTool, Message, LMRequest, Part, TextPart, build_default

lm = build_default(use_pycurl=True)

tools = (
    FunctionTool(
        name="get_weather",
        description="Get weather by city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
)

req = LMRequest(
    model="gpt-4.1-mini",
    tools=tools,
    messages=(Message.user("What is the weather in Montreal?"),),
)

resp = lm.complete(req)

tool_calls = [p for p in resp.message.parts if p.type == "tool_call"]
if tool_calls:
    call = tool_calls[0]
    result = Part.tool_result(
        id=call.id,
        content=[TextPart(text='{"temp_c": 7, "condition": "cloudy"}')],
    )
    followup = LMRequest(
        model=req.model,
        tools=tools,
        messages=(
            Message.user("What is the weather in Montreal?"),
            Message(role="assistant", parts=(call,)),
            Message(role="tool", parts=(result,)),
        ),
    )
    final = lm.complete(followup)
    print(final.message.parts[0].text)
```

## Related runnable script

- `examples/03_tools.py`
