from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import BatchRequest, Config, FileUploadRequest, LMRequest, Message, build_default


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("SKIP: GEMINI_API_KEY/GOOGLE_API_KEY not set")
        return

    lm = build_default(use_pycurl=True)
    up = lm.file_upload(FileUploadRequest(filename="notes.txt", bytes_data=b"hello", media_type="text/plain"), provider="gemini")
    print("file:", up.id)

    req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("Reply with ok"),), config=Config())
    batch = lm.batch_submit(BatchRequest(model="gemini-2.0-flash-lite", requests=(req, req)), provider="gemini")
    print("batch:", batch.id, batch.status)


if __name__ == "__main__":
    main()
