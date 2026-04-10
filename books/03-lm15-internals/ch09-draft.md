# Chapter 9: Auth

Between "the user has an API key" and "the HTTP request includes an `Authorization: Bearer sk-...` header," several small decisions happen. Where does the key come from? What format is the header? What if the provider uses a query parameter instead of a header? What if the key is in a `.bashrc` file with `export` in front of it?

`auth.py` and parts of `factory.py` handle all of this. The system has two halves: key *resolution* (finding the key) and key *application* (putting it on the HTTP request).

## Key Resolution

The priority chain, implemented in `build_default()`:

1. **`api_key=` parameter** — the user passed a key explicitly. Wins always.
2. **`env=` file** — the user pointed to a file. lm15 parses it and extracts recognized keys.
3. **`os.environ`** — ambient environment variables. Checked last.

If a key is found at any level, lower levels aren't checked for that provider. If `api_key="sk-..."` is passed for OpenAI, lm15 won't look for `OPENAI_API_KEY` in the environment.

### The api_key parameter

Three forms:

```python
# String → one key, either for the hinted provider or broadcast to all
lm15.complete("gpt-4.1-mini", "Hello.", api_key="sk-...")

# Dict → per-provider keys
lm15.complete("gpt-4.1-mini", "Hello.", api_key={
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
})
```

`_resolve_api_keys()` in `factory.py` handles this. A string with a `provider_hint` is assigned to that provider. A string without a hint is broadcast to all known providers (so a single key works if you only have one provider configured). A dict is used directly.

### The env file parser

`_parse_env_file()` handles `.env` files, but also `~/.bashrc`, `~/.zshrc`, and any file with `KEY=VALUE` lines:

```python
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[7:]
    if "=" not in line:
        continue
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip().strip("\"'")
```

It strips `export` prefixes, ignores comments and blank lines, handles both `KEY=VALUE` and `KEY="VALUE"` formats, and tolerates lines that don't match (like `alias ls='ls -la'` in a bashrc). This tolerance is deliberate — it lets users point `env=` at their shell config file instead of maintaining a separate `.env`.

The parser only extracts keys that match known provider variable names. `OPENAI_API_KEY` is extracted; `MY_CUSTOM_VAR` is ignored. The known names come from adapter manifests — each adapter declares its `env_keys` in its `ProviderManifest`:

```python
# OpenAI declares:
manifest = ProviderManifest(provider="openai", env_keys=("OPENAI_API_KEY",), ...)

# Gemini declares two possible names:
manifest = ProviderManifest(provider="gemini", env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"), ...)
```

First match wins. If both `GEMINI_API_KEY` and `GOOGLE_API_KEY` are set, the first one in the tuple is used.

### Push to os.environ

After parsing the env file for lm15's own adapters, `build_default()` also calls `_push_env_file_to_environ()`:

```python
def _push_env_file_to_environ(path, *, allowed_keys):
    # For each KEY=VALUE in the file, if KEY is in allowed_keys
    # and KEY is not already in os.environ, set it.
    os.environ.setdefault(key, value)
```

This is for third-party plugins. A plugin author's `build_adapter()` function typically does `os.getenv("MISTRAL_API_KEY")`. By pushing recognized keys from the env file into `os.environ`, lm15 ensures that plugins pick up keys from the user's `.env` file without each plugin needing to parse the file itself.

The `allowed_keys` filter prevents lm15 from setting arbitrary environment variables — only provider key names are pushed. And `setdefault` means existing environment variables aren't overwritten. Explicit environment variables win over file values.

## Key Application

Once the key is resolved, it goes to the adapter constructor:

```python
client.register(OpenAIAdapter(api_key=key, transport=transport))
```

Each adapter stores the key and applies it to HTTP requests using an auth strategy from `auth.py`:

```python
class BearerAuth(AuthStrategy):
    token: str
    def apply_headers(self, headers):
        headers["Authorization"] = f"Bearer {self.token}"
        return headers

class HeaderKeyAuth(AuthStrategy):
    header: str
    key: str
    def apply_headers(self, headers):
        headers[self.header] = self.key
        return headers

class QueryKeyAuth(AuthStrategy):
    param: str
    key: str
    def apply_params(self, params):
        params[self.param] = self.key
        return params
```

OpenAI uses `BearerAuth` — `Authorization: Bearer sk-...`. Anthropic uses `HeaderKeyAuth` — `x-api-key: sk-ant-...`. Gemini uses `QueryKeyAuth` — `?key=AIza...` in the URL.

The auth strategy is applied by the adapter when building the `HttpRequest`, before handing it to the transport. The transport sees a fully-formed request with auth headers already applied — it doesn't know or care about authentication.

## The Full Path

When you write:

```python
lm15.complete("claude-sonnet-4-5", "Hello.", env=".env")
```

Here's the full auth path:

1. `api.py` passes `env=".env"` to `build_default()`
2. `build_default()` parses `.env`, finds `ANTHROPIC_API_KEY=sk-ant-...`
3. `build_default()` creates `AnthropicAdapter(api_key="sk-ant-...", transport=...)`
4. `AnthropicAdapter` stores the key and creates `HeaderKeyAuth(header="x-api-key", key="sk-ant-...")`
5. When `complete()` is called, the adapter builds an `HttpRequest` and calls `auth.apply_headers(headers)`
6. The transport sends the request with `x-api-key: sk-ant-...` in the headers
7. Anthropic's API validates the key and returns a response

Six steps from `.env` file to HTTP header. Each step is in a different module. The separation is deliberate — key resolution is the factory's job, auth strategy is the adapter's job, header application is the auth strategy's job, and HTTP is the transport's job. No layer does two things.

The final chapter covers errors — the typed hierarchy that maps HTTP status codes to Python exceptions and propagates them cleanly through the stack.
