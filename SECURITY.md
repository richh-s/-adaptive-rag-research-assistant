# Security

## Reporting a vulnerability

Report suspected vulnerabilities by opening a [private security advisory][advisory] on this
repository. Please do not open a public issue for anything exploitable.

Include what you did, what happened, and what you expected. A proof of concept helps, but a
clear description of the flaw is worth more than a working exploit — do not run one against a
deployment you do not own.

Expect an acknowledgement within **3 working days** and an assessment within **10**. This is a
portfolio project maintained by one person, not a vendor with an on-call rotation; that is the
honest commitment rather than an aspirational SLA.

[advisory]: https://github.com/richh-s/adaptive-rag-research-assistant/security/advisories/new

## What is in scope

The API surface (`src/rag_assistant/api.py`), the authentication, identity and tenancy layers
(`auth.py`, `oidc.py`, `tenancy.py`, `ingestion/ownership.py`, `ingestion/acl.py`), the
ingestion path including URL fetch, file upload and source connectors (`connectors/`), the
web UI's sign-in flow (`frontend/src/api/sso.ts`), and the container image.

Out of scope: the behaviour of the underlying language models, denial of service by simply
sending expensive-but-valid questions (that is what the rate limiter and the per-tenant token
budget bound, and they are documented as bounds rather than guarantees), and anything
requiring an operator to have already misconfigured the deployment in a way the README warns
against.

## Threat model

Written down because a control nobody can name the attacker for is a control nobody can
evaluate. Each row names what the mitigation does *not* cover, which is the part that matters
during a review.

| Threat | Mitigation | What it does not cover |
| --- | --- | --- |
| **Cross-tenant data access** — one API key reading another tenant's documents or conversations | Retrieval, ingestion, conversations and feedback are all scoped by resolved owner; the filter is pushed into the query rather than applied to results, and a foreign conversation id 404s identically to a nonexistent one. Exercised by `tests/test_tenancy.py` and `tests/test_chroma_server.py` | Under the default `TENANT_ISOLATION=filter`, isolation is a metadata predicate, not storage, and a query-construction bug is a cross-tenant read. `strict` gives Chroma one collection per tenant; pgvector always has a fail-closed row-level-security policy (`tests/test_pgvector_isolation.py`, run as a non-superuser role) -- which Postgres does not apply to superusers or `BYPASSRLS` roles, so it is inert unless the app connects as an ordinary role. `/ready` reports which |
| **Intra-tenant data access** — a user reading a document restricted to others in their tenant | Per-document ACLs (users, groups, email domains) stamped on every chunk and enforced inside the vector and keyword searches; listings, the router's corpus description, deletion and ACL edits apply the same check; an unreadable ACL sidecar denies everyone; conversations and feedback are stored per user (`tests/test_document_acl.py`) | ACLs are string comparisons against the caller's identity: they are exactly as correct as the mapping between the source system's principals and the IdP's claims (see the README's Known limitations). A tenant-wide API key or an `OIDC_ADMIN_GROUPS` member bypasses them by design. Answers are cached by question, route and the *set of documents retrieved*, so two callers share a cached answer only when both could read every document it came from |
| **Forged or misdirected SSO tokens** | Local signature verification against the issuer's JWKS; only configured asymmetric algorithms (`none` and HS256 algorithm-confusion are tested and refused); exact issuer; required audience (refusing to start without one); required expiry; tenant claim required when configured rather than defaulted; discovery documents naming another issuer refused; JWKS refetch on unknown `kid` rate-limited (`tests/test_oidc.py`) | Token revocation: a stolen access token is valid until it expires, since tokens are verified locally rather than introspected per request. Keep access-token lifetimes short at the IdP. The browser holds its token in `sessionStorage`, which an XSS bug in the UI could read |
| **Connector credential or content misuse** — a sync publishing restricted documents, deleting a corpus on a bad listing, or leaking a credential | Secrets are referenced by environment-variable name, never stored in the config; Confluence refuses non-HTTPS base URLs; redirects are not followed with credentials; permissions are translated conservatively (multi-level Confluence restrictions intersected, unreadable Drive permissions deny everyone), and connectors that cannot see the full permission model require an explicit `default_acl`; deletion sync is refused beyond `CONNECTOR_MAX_DELETE_FRACTION`; symlinks out of a synced share are not followed | The service account's own access bounds what can be synced, and anything it can read is indexed into the configured tenant. Space-level Confluence permissions are declared by the operator, not read |
| **Prompt injection via ingested documents or web results** | All five prompts that interpolate untrusted text -- synthesis, grading, routing, condensation and the groundedness check -- fence it with a per-request nonce and state the trust hierarchy first; a test fails the build if a sixth is added without one. Attempts are counted, never silently dropped (`content_trust.py`) | No prompt-level defense can guarantee a model obeys. The tests assert the mechanism, not model compliance. What actually bounds the blast radius: synthesis has no tools, retrieval is tenant-scoped, and citations are built from the pipeline's own selections rather than from model claims. The groundedness check is the highest-value target in the pipeline -- a document that talks it into marking every claim supported turns its score into a rubber stamp, which is worse than no check because the number is still reported |
| **Credential disclosure** — API keys in logs, traces or error responses | Keys are compared with `hmac.compare_digest` and only ever logged as a truncated SHA-256 fingerprint; `API_KEYS_FILE` keeps secrets out of the process listing; `.env` is git-ignored and secret scanning runs in CI | A key with `write` scope can do everything a write endpoint allows. There is no per-key audit of *what* was retrieved, only that a key was used |
| **Server-side request forgery** via `POST /api/v1/ingest/url` | The fetcher restricts scheme, resolves and rejects private/loopback/link-local addresses, caps response size and follows a bounded redirect chain (`ingestion/url_fetch.py`) | DNS rebinding between the check and the fetch is not defended against. A deployment that needs to fetch from inside a private network has to relax this, and then owns the consequence |
| **Path traversal via upload filename** | Upload stems are stripped to `[A-Za-z0-9_-]` before touching the corpus directory (`_safe_stem`) | Nothing: the tracked regression test exists because this bug was real once |
| **Resource exhaustion** — upload size, question length, runaway graph cost | 25MB upload cap enforced while streaming, 2000-char question cap, bounded graph timeout, per-caller and global rate limits, per-tenant daily token budget checked before the run | Rate limits are per-replica unless `RATE_LIMIT_STORAGE_URI` is shared — see the README. Token spend on embeddings and vision is estimated, not reported by the provider |
| **Personal data reaching the index** — an upload carrying contact details, payment data or credentials, embedded and then quotable in an answer | Ingested text is scanned for emails, phone numbers, Luhn-valid card numbers, US SSNs, IBANs and cloud access keys. `PII_MODE=flag` (default) counts and logs; `redact` replaces each match with a category marker before the text is embedded, keyword-indexed or stored as a parent section, so it cannot be retrieved, cited or quoted (`pii.py`) | Regexes find formats, not people: names, addresses, dates of birth and national IDs without fixed structure pass through. Redaction applies only to what is ingested after it is enabled (`ingest --full` to reapply) and never touches the uploaded file on disk, which stays readable. Conversation transcripts are not scanned and are stored in plaintext |
| **Destructive action by an under-privileged key** — a read-only credential deleting indexed data | `DELETE /api/v1/sources/{source}` and `DELETE /api/v1/tenant/data` both require the `write` scope, and deletion is scoped to sources the caller owns and may read; a source belonging to another tenant, or restricted from the caller, reports as absent rather than forbidden, so the endpoint cannot be used to probe for filenames. Rebuilding the index and switching generations need a separate `admin` scope that no SSO token carries, and are refused outright when auth is disabled | Any key with `write` scope can delete that tenant's documents; there is no separate delete scope and no undo. Recovery is the backup archive |
| **Supply chain** — a vulnerable or malicious dependency | `uv.lock` pins every transitive dependency by hash; `pip-audit`, `npm audit`, image scanning and secret scanning run in CI | The audit jobs are `continue-on-error`: they warn, they do not block. Promote them once someone owns triage. No SBOM is published and images are not signed |

## What has not been done

An independent security review. Everything above is the author's own reasoning about the
author's own code, plus static analysis, which is a category of evidence that cannot find what
it was not taught to look for. No penetration test has been performed against a deployment.

Treat the table as a description of intent and of the tests that exist, not as an assurance
that the intent is achieved.
