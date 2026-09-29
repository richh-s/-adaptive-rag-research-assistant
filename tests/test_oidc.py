"""Single sign-on: verifying identity-provider tokens.

Signed with a real RSA key generated per test run and verified against a JWKS document the
test serves, so every check here exercises real signature verification. Each rejection test
corresponds to a known way JWT validation goes wrong in deployed systems: the wrong audience,
an unsigned token, a symmetric algorithm keyed with the public key, a missing tenant.
"""

import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from rag_assistant import api, oidc

ISSUER = "https://login.example.com/tenant-1"
AUDIENCE = "api://rag-assistant"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def idp(monkeypatch, signing_key):
    """An identity provider: a discovery document and a JWKS with one key, served through a
    patched httpx.get that counts requests."""
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public_jwk.update(kid="key-1", use="sig", alg="RS256")
    calls = {"discovery": 0, "jwks": 0}

    def _get(url, **kwargs):
        if url.endswith("/.well-known/openid-configuration"):
            calls["discovery"] += 1
            body = {"issuer": ISSUER, "jwks_uri": "https://login.example.com/jwks"}
        elif url == "https://login.example.com/jwks":
            calls["jwks"] += 1
            body = {"keys": [public_jwk]}
        else:
            raise AssertionError(f"unexpected fetch {url}")
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(oidc.httpx, "get", _get)
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("OIDC_WRITE_GROUPS", "editors")
    monkeypatch.setenv("OIDC_ADMIN_GROUPS", "rag-admins")
    return calls


def _token(signing_key, kid="key-1", algorithm="RS256", **overrides):
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "00u-dana",
        "email": "dana@acme.com",
        "groups": ["finance"],
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, signing_key, algorithm=algorithm, headers={"kid": kid})


def test_a_valid_token_becomes_a_principal(idp, signing_key):
    principal = oidc.verify_token(_token(signing_key))

    assert principal.method == "oidc"
    assert principal.owner == "public"  # no tenant claim configured -> OIDC_DEFAULT_TENANT
    assert principal.subject == "00u-dana"
    assert principal.groups == {"finance"}
    assert principal.scopes == {"read"}
    assert principal.principals() == {
        "user:00u-dana",
        "user:dana@acme.com",
        "domain:acme.com",
        "group:finance",
    }


def test_write_comes_from_a_write_group_or_the_write_scope(idp, signing_key):
    assert oidc.verify_token(_token(signing_key, groups=["editors"])).has_scope("write")
    machine = oidc.verify_token(_token(signing_key, groups=None, scope="openid rag.write"))
    assert machine.has_scope("write")


def test_an_admin_group_bypasses_document_acls_but_is_not_a_deployment_admin(idp, signing_key):
    """A tenant's admins may read everything in their tenant; they may not rebuild the index
    every tenant shares."""
    principal = oidc.verify_token(_token(signing_key, groups=["rag-admins"]))

    assert principal.principals() is None
    assert principal.has_scope("write")
    assert not principal.has_scope("admin")


def test_the_tenant_comes_from_the_configured_claim(idp, signing_key, monkeypatch):
    monkeypatch.setenv("OIDC_TENANT_CLAIM", "org.id")
    principal = oidc.verify_token(_token(signing_key, org={"id": "acme corp"}))
    assert principal.owner == "acme_corp"


def test_a_token_missing_the_tenant_claim_is_rejected_not_defaulted(idp, signing_key, monkeypatch):
    """Defaulting would file the user under the default tenant and show them its documents."""
    monkeypatch.setenv("OIDC_TENANT_CLAIM", "org_id")
    with pytest.raises(oidc.TokenRejected, match="org_id"):
        oidc.verify_token(_token(signing_key))


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"aud": "api://some-other-app"}, "audience"),
        ({"iss": "https://evil.example.com"}, "issuer"),
        ({"exp": int(time.time()) - 3600}, "expired"),
        ({"exp": None}, "exp"),
        ({"sub": None}, "sub"),
    ],
)
def test_tokens_failing_a_claim_check_are_rejected(idp, signing_key, overrides, reason):
    with pytest.raises(oidc.TokenRejected):
        oidc.verify_token(_token(signing_key, **overrides))


def test_a_token_signed_by_another_key_is_rejected(idp):
    impostor = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(oidc.TokenRejected):
        oidc.verify_token(_token(impostor))


def test_an_unsigned_token_is_rejected(idp):
    now = int(time.time())
    unsigned = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": now + 600},
        key=None,
        algorithm="none",
    )
    with pytest.raises(oidc.TokenRejected, match="algorithm"):
        oidc.verify_token(unsigned)


def test_a_symmetric_token_keyed_with_the_public_key_is_rejected(idp, signing_key):
    """The classic algorithm-confusion attack: sign HS256 using the RSA *public* key, which
    anyone can download, and hope the verifier uses it as an HMAC secret."""
    from cryptography.hazmat.primitives import serialization

    public_pem = signing_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    now = int(time.time())
    # PyJWT refuses to HMAC-sign with a PEM public key, which is itself a defence; the token
    # is built by hand to confirm the verifier refuses it as well.
    import base64
    import hashlib
    import hmac

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "HS256", "kid": "key-1", "typ": "JWT"}).encode())
    payload = b64(
        json.dumps({"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": now + 600}).encode()
    )
    signature = b64(hmac.new(public_pem, f"{header}.{payload}".encode(), hashlib.sha256).digest())
    forged = f"{header}.{payload}.{signature}"

    with pytest.raises(oidc.TokenRejected, match="algorithm"):
        oidc.verify_token(forged)


def test_signing_keys_are_cached_and_unknown_kids_cannot_force_refetches(idp, signing_key):
    """Every request re-fetching the JWKS would make the identity provider a per-request
    dependency, and a stream of tokens with random kids would turn into outbound traffic."""
    for _ in range(5):
        oidc.verify_token(_token(signing_key))
    for index in range(5):
        with pytest.raises(oidc.TokenRejected):
            oidc.verify_token(_token(signing_key, kid=f"random-{index}"))

    assert idp["jwks"] == 1
    assert idp["discovery"] == 1


def test_a_discovery_document_for_another_issuer_is_refused(monkeypatch, signing_key):
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(
        oidc.httpx,
        "get",
        lambda url, **k: httpx.Response(
            200,
            json={"issuer": "https://evil.example.com", "jwks_uri": "https://evil.example.com/k"},
            request=httpx.Request("GET", url),
        ),
    )
    with pytest.raises(oidc.TokenRejected):
        oidc.verify_token(_token(signing_key))


def test_an_issuer_without_an_audience_refuses_to_start(monkeypatch):
    from rag_assistant.config import get_settings

    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.delenv("OIDC_AUDIENCE", raising=False)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="OIDC_AUDIENCE"):
        get_settings()


# ---- through the API ----


@pytest.fixture
def client(idp, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPM", "1000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "1000")
    return TestClient(api.app)


def test_a_bearer_token_authenticates_an_api_request(client, signing_key):
    response = client.get(
        "/api/v1/auth/check", headers={"Authorization": f"Bearer {_token(signing_key)}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "oidc"
    assert body["user"] == "dana@acme.com"
    assert body["groups"] == ["finance"]


def test_an_invalid_token_gets_a_generic_401(client, signing_key):
    response = client.get(
        "/api/v1/auth/check",
        headers={"Authorization": f"Bearer {_token(signing_key, aud='api://other')}"},
    )

    assert response.status_code == 401
    # Which check failed is for the audit log, not for the caller.
    assert "audience" not in response.text.lower()


def test_a_read_only_user_cannot_upload(client, signing_key):
    response = client.post(
        "/api/v1/ingest",
        headers={"Authorization": f"Bearer {_token(signing_key)}"},
        files={"file": ("x.md", b"hello", "text/markdown")},
    )
    assert response.status_code == 403


def test_api_keys_keep_working_when_sso_is_turned_on(client, monkeypatch):
    monkeypatch.setenv("API_KEYS", "ops:sk-ops")
    response = client.get("/api/v1/auth/check", headers={"X-API-Key": "sk-ops"})
    assert response.status_code == 200
    assert response.json()["method"] == "api_key"


def test_the_sign_in_config_is_public_and_contains_nothing_secret(client, monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_ID", "spa-client")
    response = client.get("/auth/config")

    assert response.status_code == 200
    assert response.json()["oidc"] == {
        "issuer": ISSUER,
        "client_id": "spa-client",
        "scopes": "openid profile email",
        "audience": AUDIENCE,
    }


def test_admin_endpoints_are_closed_in_open_demo_mode(monkeypatch):
    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    response = TestClient(api.app).get("/api/v1/admin/index")
    assert response.status_code == 403


def test_admin_endpoints_need_the_admin_scope(client, signing_key, tmp_path, monkeypatch):
    keys = tmp_path / "keys.json"
    keys.write_text(
        json.dumps(
            {
                "keys": [
                    {"key": "sk-op", "owner": "ops", "scopes": ["read", "write", "admin"]},
                    {"key": "sk-rw", "owner": "ops"},
                ]
            }
        )
    )
    monkeypatch.setenv("API_KEYS_FILE", str(keys))

    admin_token = _token(signing_key, groups=["rag-admins"])
    assert (
        client.get(
            "/api/v1/admin/index", headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code
        == 403
    )
    assert client.get("/api/v1/admin/index", headers={"X-API-Key": "sk-rw"}).status_code == 403
    assert client.get("/api/v1/admin/index", headers={"X-API-Key": "sk-op"}).status_code == 200
