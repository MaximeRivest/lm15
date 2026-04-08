# Cookbook 05 — Files and Batches

## Goal

Use normalized file upload and batch submission APIs.

## Example

```python
from lm15 import BatchRequest, Config, FileUploadRequest, LMRequest, Message, build_default

lm = build_default(use_pycurl=True)

# 1) Upload file
uploaded = lm.file_upload(
    FileUploadRequest(filename="notes.txt", bytes_data=b"hello world", media_type="text/plain"),
    provider="gemini",
)
print("file id:", uploaded.id)

# 2) Submit batch
r = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("Reply with: ok"),), config=Config())
batch = lm.batch_submit(BatchRequest(model="gemini-2.0-flash-lite", requests=(r, r)), provider="gemini")
print("batch:", batch.id, batch.status)
```

## Notes

- Provider-native batch lifecycle APIs differ.
- LM15 currently normalizes submit path and exposes provider payload in `BatchResponse.provider`.

## Related runnable script

- `examples/05_files_batches.py`
