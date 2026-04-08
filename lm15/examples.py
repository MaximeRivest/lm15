from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lm15 import LMRequest, LMResponse, Message, Part, UniversalLM, Usage


@dataclass(slots=True)
class EchoAdapter:
    provider: str = "echo"

    def complete(self, request: LMRequest) -> LMResponse:
        text = "\n".join(p.text or "" for m in request.messages for p in m.parts if p.type == "text")
        return LMResponse(
            id="echo-1",
            model=request.model,
            message=Message(role="assistant", parts=(Part.text_part(text),)),
            finish_reason="stop",
            usage=Usage(),
            provider={"echo": True},
        )

    def stream(self, request: LMRequest):
        yield from ()


if __name__ == "__main__":
    lm = UniversalLM()
    lm.register(EchoAdapter())
    req = LMRequest(model="echo-1", messages=(Message.user("hello"),))
    print(lm.complete(req, provider="echo").message.parts[0].text)
