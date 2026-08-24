"""Fetches a public web page and converts it into a corpus-ready markdown document.

The security posture here is deliberate: this endpoint makes the *server* issue HTTP requests
to caller-chosen URLs, which is the textbook SSRF setup -- so every hostname (including each
redirect hop's) is resolved and checked against private/loopback/link-local ranges before it
is fetched, and the response body is size-capped as it streams in.
"""

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from rag_assistant.ingestion.loaders import extract_html_text

logger = logging.getLogger(__name__)

MAX_CONTENT_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 15.0

_TEXTUAL_CONTENT_TYPES = ("text/html", "application/xhtml", "text/plain", "text/markdown")


class UrlIngestError(ValueError):
    """Raised for any user-actionable fetch problem (blocked host, non-HTML content, too
    large, unreachable) -- the API layer maps this to a 400 with the message as detail."""


@dataclass
class FetchedPage:
    url: str
    title: str | None
    text: str


def _assert_public_host(url: str) -> None:
    host = urlparse(url).hostname
    if not host:
        raise UrlIngestError(f"URL has no hostname: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlIngestError(f"Could not resolve host {host!r}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise UrlIngestError(
                f"Refusing to fetch {host!r}: it resolves to a private or local address."
            )


def fetch_page(url: str) -> FetchedPage:
    """Fetch one public http(s) page and extract its title and visible text.

    Redirects are followed manually (not via httpx's follow_redirects) because each hop's
    hostname needs the same private-address check as the original URL -- an attacker's
    public page 302-ing to http://169.254.169.254/ must be caught, not followed.
    """
    current_url = url
    with httpx.Client(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _assert_public_host(current_url)
            try:
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise UrlIngestError("Server sent a redirect with no location.")
                        current_url = str(httpx.URL(current_url).join(location))
                        if not current_url.lower().startswith(("http://", "https://")):
                            raise UrlIngestError("Redirected to a non-http(s) URL; refusing.")
                        continue

                    if response.status_code >= 400:
                        raise UrlIngestError(
                            f"Fetching {current_url!r} failed with HTTP {response.status_code}."
                        )

                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not content_type.startswith(_TEXTUAL_CONTENT_TYPES):
                        raise UrlIngestError(
                            f"Unsupported content type {content_type!r} -- only HTML/text "
                            "pages can be ingested by URL (upload PDFs as files instead)."
                        )

                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_CONTENT_BYTES:
                            raise UrlIngestError(
                                f"Page exceeds the {MAX_CONTENT_BYTES // (1024 * 1024)}MB limit."
                            )
            except httpx.HTTPError as exc:
                raise UrlIngestError(f"Could not fetch {current_url!r}: {exc}") from exc

            html = body.decode(response.encoding or "utf-8", errors="replace")
            if content_type.startswith(("text/plain", "text/markdown")):
                title, text = None, html
            else:
                title, text = extract_html_text(html)
            if not text.strip():
                raise UrlIngestError("The page contained no extractable text.")
            return FetchedPage(url=current_url, title=title, text=text)

    raise UrlIngestError(f"Too many redirects while fetching {url!r}.")


def page_to_markdown(page: FetchedPage) -> str:
    """Render the fetched page as the markdown file that lands in corpus_dir -- title as
    heading, source URL preserved in the body so answers citing this document can be traced
    back to where the content came from."""
    heading = page.title or urlparse(page.url).netloc
    return f"# {heading}\n\nSource URL: {page.url}\n\n{page.text}\n"
