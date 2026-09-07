# Cloud hosts: AWS, Azure, Google Cloud

Status: implemented 2026-09-03 from provider documentation and the cloud
SDKs' own resolver sources (spec/auth.md AUTH-1, AUTH-10, AUTH-11;
`changes/2026-09-03-cloud-hosts.md`). Live status is per door: Azure
OpenAI and Bedrock Chat have receipts; other rows remain documentation-
evidenced until their own change entry says otherwise.

A cloud host is a door to a model you already know. The wire is the same
(Anthropic Messages, OpenAI Responses or Chat Completions, Gemini); the
door changes the URL, the signing, a few body rewrites, and the
identity. You send the same `Request` and read the same `Response`.

## The doors

| Provider string | Wire | Needs | Credential |
|---|---|---|---|
| `azure:<deployment>` | OpenAI Responses | `AZURE_OPENAI_RESOURCE` | `AZURE_OPENAI_API_KEY` or the Azure chain |
| `azure-chat:<deployment>` | OpenAI Chat Completions | same | same |
| `azure-anthropic:<model>` | Anthropic Messages (Foundry) | `ANTHROPIC_FOUNDRY_RESOURCE` | `ANTHROPIC_FOUNDRY_API_KEY` or the Azure chain |
| `aws-anthropic:<model>` | Anthropic Messages (Claude Platform on AWS) | `AWS_REGION`, `ANTHROPIC_AWS_WORKSPACE_ID` | `ANTHROPIC_AWS_API_KEY` or the AWS chain (SigV4) |
| `bedrock-anthropic:anthropic.<model>` | Anthropic Messages (Bedrock, Opus 4.7+) | `AWS_REGION` | `AWS_BEARER_TOKEN_BEDROCK` or the AWS chain (SigV4) |
| `bedrock-chat:<model id>` | OpenAI Chat Completions (Bedrock runtime; versioned ids) | `AWS_REGION` | same |
| `bedrock-mantle-chat:<model id>` | OpenAI Chat Completions (Bedrock mantle; un-versioned ids; `list_models()` works) | `AWS_REGION` | same |
| `vertex:<model>` | Gemini | `GOOGLE_CLOUD_PROJECT` (`GOOGLE_CLOUD_LOCATION`, default `global`) | the Google chain |
| `vertex-anthropic:<model>` | Anthropic Messages (rawPredict) | same | same |
| `vertex-express:<model>` | Gemini | nothing | `GOOGLE_API_KEY` |

On Azure the model string is the **deployment name**; an unknown one is
`UnsupportedModelError` (HTTP 404 `DeploymentNotFound`). `list_models()`
on `azure`/`azure-chat` returns the resource's model *catalog* (every
model Azure could deploy there), not the deployments you can call. `region`
and `resource` have no default: a wrong-region default is a residency bug,
so lm15 raises `NotConfiguredError` naming the variable.

## Azure live status and exact pending work (2026-09-04)

`azure` (Responses) is live-verified for complete, stream, reasoning
(`gpt-5-mini`, including nonzero reasoning tokens), prompt-cache hits,
content-filter errors and completion stops, model listing, Files, Batch,
text-to-speech, and Realtime text WebSockets. `azure-chat` is live-
verified for complete, stream, reasoning, caching, filtering and models.
Sync and async complete/stream both ran live. The Azure chain ran live by
client secret, certificate, token-provider callable, and its `az` rung.
See `lm15-contract/changes/2026-09-04-azure-live.md` and
`2026-09-04-azure-chat-live.md`.

Two quota-gated proofs remain:

| Pending proof | Blocker | After Azure removes it |
|---|---|---|
| `azure` image generation | `gpt-image-1-mini` GlobalStandard Requests-Per-Minute quota is 0 in eastus2; quota request submitted | Rerun `research/cloud-hosts/azure/provision.sh`; it creates deployment `gpt-image-1-mini`. Run `python3 research/providers/azure/capture.py --only image --force`, review the image case, then set `AZURE.supports.images` and the support-matrix cell true. |
| `azure-anthropic` successful inference | Claude Haiku 4.5 quota is 0. Microsoft denied eastus2 because that region has no capacity. Host, `x-api-key`, both Entra scopes, secret/certificate auth, `/models` refusal, and error mapping are live-proven; no 200 inference exists. | Obtain 10 capacity units in any listed region. Run the provision command below with that region, then run the full `azure-anthropic` capture, draft/review goldens, and run every harness direction. |

Post-quota Claude command (use the region Azure grants):

```bash
FDY_LOC=westus3 \
LM15_LAB_ORG='lm15-dev (open-source project, Maxime Rivest)' \
LM15_LAB_INDUSTRY='Software & Internet' \
LM15_LAB_COUNTRY=CA \
bash research/cloud-hosts/azure/provision.sh

python3 research/providers/azure-anthropic/capture.py --force
```

Sora/video is deliberately excluded, not pending: the user chose to skip
`sora-2` because the product is going away and video is not a priority.
Azure embeddings answered on `/openai/v1`; transcription answered on
Azure's deployment-scoped `?api-version=2025-04-01-preview` route (the v1
route returned 404). lm15 has no canonical embedding or transcription
surface; both are outside the current library API rather than incomplete
Azure implementations.

```python
from lm15 import LMRouter, Request, Message

router = LMRouter()  # reads AWS_REGION and the AWS chain from the environment
response = router.complete(Request(
    model="bedrock-anthropic:anthropic.claude-opus-5",
    messages=[Message.user("Say ok.")],
))
```

Settings can be passed instead of read from the environment:

```python
from lm15.router import RouterConfig

router = LMRouter(RouterConfig(settings={"bedrock-anthropic": {"region": "us-east-1"}}))
```

## Which identity is used

The same one the cloud's own SDK would pick on the same machine. lm15
walks boto3's, `DefaultAzureCredential`'s and google-auth's default
chains in their exact order, with an explicit `api_keys` entry first.
The doctor shows the walk without touching the network:

```python
from lm15.doctor import explain_auth
print(explain_auth("bedrock-anthropic"))                 # from the environment
print(explain_auth("azure", config=router.config))       # what THIS router will do: its keys and settings
```

```
auth for provider 'bedrock-anthropic':
   - explicit api_keys entry: not provided
   - env $AWS_BEARER_TOKEN_BEDROCK: not set
   - env $AWS_ACCESS_KEY_ID (+SECRET, +SESSION_TOKEN): not set
   - profile assume-role via STS: profile 'default' has no role_arn with a source
   ? web identity via STS: token file /var/run/secrets/eks.amazonaws.com/serviceaccount/token → arn:aws:iam::…:role/app (STS call at request time)
   …
   ? EC2 instance metadata (IMDSv2): instance metadata probed at request time
  configured: probably — web identity via STS, EC2 instance metadata (IMDSv2) (unprobed offline)
  setting region: eu-west-1
```

`?` marks a rung the offline doctor cannot decide: it needs the network
or a subprocess, and its configuration is present. Such a rung runs
first at request time and may win over a later `=>` rung.

The chains, rung by rung, are in `spec/auth.md` AUTH-1. Rungs that need
signing use the standard library only: SigV4 is HMAC-SHA256; the Google
service-account and Entra certificate assertions are RS256 signed by a
pure-Python signer (about 75 ms per signature, once per one-hour token).

## Credentials are values, not strings

`api_key=` accepts a string (an API key), an `lm15.credentials` value,
or a zero-argument callable returning either:

```python
from lm15.credentials import AwsCredentials, BearerToken
from lm15.providers import AnthropicLM
from lm15 import access

lm = AnthropicLM(
    api_key=AwsCredentials("AKIA…", "…", session_token="…"),
    access=access.BEDROCK_ANTHROPIC,
    settings={"region": "us-east-1"},
)
```

A door lists the schemes it accepts; the credential kind picks one. AWS
credentials always travel as a SigV4 signature; a bearer token under
`Authorization`, or under the door's key header on the two doors that
carry their tokens there (`bedrock-anthropic`, `aws-anthropic`: a
Bedrock short-term key goes as `x-api-key`); an API key under the door's
key header (`x-api-key`, `api-key`) or, for Vertex express, as `?key=`.
The wrong kind for a door fails at construction.

A Bedrock short-term API key (`AWS_BEARER_TOKEN_BEDROCK`) works on both
Bedrock doors (live 2026-09-04 on `bedrock-chat`). lm15 does not mint
one: with AWS credentials it signs each request directly, which is what
the key would have done for you. Mint one only for a tool that speaks
bearer only (AWS's `aws-bedrock-token-generator`, or
`lm15-contract/research/providers/_aws_bearer.py`).

## What is not supported, and what to do instead

Each is a `NotConfiguredError` with the fix in the message; none falls
through silently.

- `aws login` sessions are read from their cache while fresh; refresh
  needs a DPoP-bound key that lm15 does not implement — run `aws login`.
- Windows-only and interactive Azure rungs (Visual Studio shared cache,
  VS Code, browser, broker) and username/password — run `az login`.
- Azure Service Fabric managed identity (TLS thumbprint pinning) — use a
  certificate, a secret, or another managed-identity host.
- Encrypted PEM keys and PKCS#12 certificates — `openssl pkey` /
  `openssl pkcs12 -nodes` first.
- Google `external_account` with an AWS credential source, and the
  `external_account_authorized_user` / `gdch_service_account` types —
  use a file, URL or executable source, or a service account.
- Bedrock's native Converse wire and the binary event-stream framing
  (Nova, Llama, Mistral on Bedrock) — phase 2; `bedrock:` names nothing
  yet.

## Security and lifecycle limits

Credential files, endpoint overrides, and subprocess configuration are trusted
inputs. Do not accept them from an untrusted tenant. Credential HTTP requests
do not follow redirects. Error messages omit command output and token response
bodies. Request and credential reprs hide secret-bearing fields.

Cloud chains cache expiring credentials until the five-minute refresh window.
A refreshed credential that is already expired raises `AuthError`. CLI bearer
tokens without expiry metadata are resolved again on the next request, rather
than cached forever. No cloud credential cache is written to disk.

Cloud credential providers are synchronous. A refresh can block an async
adapter's event loop. Use a pre-refreshed credential provider when that latency
is unacceptable.

The stdlib RSA signer uses variable-time Python integer arithmetic, without
blinding. It is not hardened against timing attacks. Use an external credential
provider when that threat matters. Encrypted-key support and a hardened crypto
backend need a separate dependency decision.

## Per-door refusals

A feature the door does not carry raises rather than being dropped
(MAP-8). The documented lists are in `spec/auth.md` AUTH-10; they are
confirmed cell by cell when each door is captured live.
