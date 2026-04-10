# Chapter 7: Middleware

Where do retries happen? Not in the adapter — it shouldn't know about retry policy. Not in the transport — it shouldn't know about requests. Not in the Model — it shouldn't know about HTTP failures. Not in the sugar — it delegates immediately.

The answer is middleware. It sits between the client and the adapter, wrapping the call so it can be intercepted, retried, cached, or observed.

## The Middleware Contract

A middleware is a function with this signature:

```python
def middleware(request: LMRequest, next_fn: CompleteFn) -> LMResponse:
```

`request` is the incoming request. `next_fn` is a callable that represents "everything downstream" — the next middleware in the chain, or the adapter if this is the last middleware. You call `next_fn(request)` to proceed, and you can do things before, after, or instead of that call.

This is a standard pattern — Express.js middleware, Django middleware, Rack middleware all work the same way. The power is in composition: each middleware wraps the next, creating a nested call stack.

## The Pipeline

`MiddlewarePipeline` in `middleware.py` manages two lists — `complete_mw` for blocking calls and `stream_mw` for streaming calls. The `wrap_complete()` method builds the call chain:

```python
def wrap_complete(self, fn: CompleteFn) -> CompleteFn:
    wrapped = fn
    for mw in reversed(self.complete_mw):
        prev = wrapped
        wrapped = lambda req, mw=mw, prev=prev: mw(req, prev)
    return wrapped
```

It iterates the middleware list in reverse, wrapping each one around the previous. The result is a single callable that, when invoked, runs middleware[0] first, then middleware[1], and so on, with the original adapter function at the bottom.

`UniversalLM.complete()` calls `self.middleware.wrap_complete(adapter.complete)` and then calls the result with the request. The middleware chain runs, the adapter gets called somewhere in the middle, and the response propagates back up through the chain.

## Built-In Middleware

lm15 ships three middleware functions. Each one is a factory that returns a middleware closure.

### with_retries

```python
def with_retries(max_retries: int = 2, sleep_base: float = 0.2) -> CompleteMiddleware:
    transient = (RateLimitError, TimeoutError, ServerError, TransportError)
    
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        for i in range(max_retries + 1):
            try:
                return nxt(req)
            except transient as e:
                if i == max_retries:
                    raise
                time.sleep(sleep_base * (2**i))
    return middleware
```

Fifteen lines. Catch transient errors, wait with exponential backoff, retry. After `max_retries` failures, raise the last exception. The retry classification is explicit: `RateLimitError`, `TimeoutError`, `ServerError`, and `TransportError` are retryable. `AuthError` and `InvalidRequestError` are not.

Note: `Model`'s own retry logic (`retries=` parameter) uses the same transient error classification but is implemented separately — it wraps `_complete_with_cache()`, which is a different point in the stack than middleware. The two retry mechanisms don't interact; you'd use one or the other, not both.

### with_cache

```python
def with_cache(cache: dict[str, LMResponse]) -> CompleteMiddleware:
    def key(req: LMRequest) -> str:
        return str((req.model, req.system, req.messages, req.tools, req.config))
    
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        k = key(req)
        if k in cache:
            return cache[k]
        resp = nxt(req)
        cache[k] = resp
        return resp
    return middleware
```

Even simpler. Hash the request (using `str()` on a tuple of its fields — crude but effective since all fields are frozen and thus have stable string representations). If it's in the cache, return it without calling downstream. If not, call downstream, cache the result, return it.

The cache dict is passed in by the caller, which means the caller controls the cache's lifetime and scope. Pass an empty dict for a request-local cache. Pass a module-level dict for a persistent cache. The middleware doesn't manage eviction — if you need LRU or TTL, use a `cachetools` dict or similar.

### with_history

```python
def with_history(history: list[dict[str, Any]]) -> CompleteMiddleware:
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        started = time.time()
        resp = nxt(req)
        history.append({
            "ts": started,
            "model": req.model,
            "messages": len(req.messages),
            "finish_reason": resp.finish_reason,
            "usage": { ... },
        })
        return resp
    return middleware
```

Records metadata about every call — timestamp, model, message count, finish reason, token usage — into a list you provide. This is the v1 observability system. Book 1's `Model.history` serves a similar purpose at a higher level, but `with_history` captures every call through the middleware pipeline, including retries and cache misses.

## Composition and Ordering

Middleware order matters. If you add cache before retries:

```python
pipeline.complete_mw.append(with_cache(cache))
pipeline.complete_mw.append(with_retries(max_retries=2))
```

The call chain is: cache → retries → adapter. A cached response bypasses retries entirely (good — no need to retry a cache hit). A cache miss goes through retries (good — transient failures get retried). A successful retry gets cached (good — the recovered response is stored).

If you reverse the order — retries before cache — then a cache miss is retried even though it's not a transient failure, and a cached response goes through the retry wrapper unnecessarily. Order matters.

The general principle: **read-only middleware (cache, logging) outside; side-effecting middleware (retries, transforms) inside.** "Outside" means earlier in the list, which means it wraps everything downstream.

## Writing Custom Middleware

The pattern is always the same:

```python
def my_middleware_factory(**config) -> CompleteMiddleware:
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        # Before the call: inspect/modify the request
        # ...
        resp = nxt(req)  # Call downstream
        # After the call: inspect/modify the response
        # ...
        return resp
    return middleware
```

A token budget middleware:

```python
def with_token_budget(budget: int) -> CompleteMiddleware:
    total = [0]  # mutable counter in closure
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        if total[0] >= budget:
            raise RuntimeError(f"Token budget exceeded: {total[0]} >= {budget}")
        resp = nxt(req)
        total[0] += resp.usage.total_tokens
        return resp
    return middleware
```

A model fallback middleware:

```python
def with_fallback(fallback_model: str) -> CompleteMiddleware:
    def middleware(req: LMRequest, nxt: CompleteFn) -> LMResponse:
        try:
            return nxt(req)
        except ProviderError:
            fallback_req = LMRequest(
                model=fallback_model, messages=req.messages,
                system=req.system, tools=req.tools, config=req.config)
            return nxt(fallback_req)
    return middleware
```

These are the v1 (low-level) API. Most users will never write middleware — `Model`'s `retries=` and `cache=` parameters handle the common cases. But middleware is how you extend the behavior of the call pipeline without modifying any existing code, and plugin authors use it to add cross-cutting concerns to adapters they don't control.

The next chapter covers discovery — how `lm15.models()` builds its list of available models by querying live APIs, merging with fallback catalogs, and filtering by capabilities.
