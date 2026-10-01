"""Bounded HTTPS fetches, allowlisted hosts, pinned public IP and TLS hostname checks."""
import http.client
import ipaddress
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from .models import Download, IngestionError

ALLOWED_HOSTS = frozenset({"www.fca.org.uk", "fca.org.uk", "handbook.fca.org.uk", "www.handbook.fca.org.uk", "api-handbook.fca.org.uk",
                           # Türkiye: the consolidated legislation service and the Official Gazette.
                           "www.mevzuat.gov.tr", "mevzuat.gov.tr", "www.resmigazete.gov.tr", "resmigazete.gov.tr"})
MAX_BYTES = 12 * 1024 * 1024
# The gov.tr servers omit their intermediate CA certificate from the TLS handshake. A browser
# fetches it on the fly; this fetcher does not, so the intermediate ships with the package
# (provenance in the file header) and is added to the system trust store. Verification stays
# on: the chain still has to end at a root the operating system trusts.
EXTRA_CA = Path(__file__).with_name("certs") / "geotrust-tls-rsa-ca-g1.pem"


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if EXTRA_CA.exists():
        context.load_verify_locations(cafile=str(EXTRA_CA))
    return context


def validate_url(url: str) -> str:
    try:
        value = urlsplit(url)
        valid = (value.scheme == "https" and value.hostname in ALLOWED_HOSTS
                 and value.port in (None, 443) and not value.username and not value.password)
    except ValueError:
        valid = False
    if not valid or len(url) > 2048 or any(ord(c) < 33 for c in url) or "\\" in url:
        raise IngestionError("Only HTTPS URLs on explicitly allowed regulatory hosts are accepted")
    return urlunsplit(("https", value.hostname, value.path or "/", value.query, ""))


def public_address(host: str) -> str:
    addresses = [row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
    if not addresses or any(not ipaddress.ip_address(addr).is_global for addr in addresses):
        raise IngestionError("Source resolved to a non-public address")
    return addresses[0]


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        address = public_address(self.host)
        raw = socket.create_connection((address, self.port), self.timeout)
        try:
            # The connected IP is the validated IP; DNS is not resolved a second time.
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class RetryableFetch(IngestionError):
    pass


def _download(url: str, body: bytes | None = None) -> Download:
    original = current = validate_url(url)
    deadline = time.monotonic() + 60
    for _ in range(6):
        parsed = urlsplit(current)
        connection = PinnedHTTPSConnection(parsed.hostname, timeout=15, context=tls_context())
        try:
            path = urlunsplit(("", "", parsed.path, parsed.query, ""))
            headers = {"User-Agent": "RegChain/0.2 (regulatory-source-ingestion)",
                       "Accept": "text/html, application/pdf, application/json", "Accept-Encoding": "identity"}
            if body is None:
                connection.request("GET", path, headers=headers)
            else:
                # A search query to a public catalogue; never company or policy content.
                connection.request("POST", path, body=body, headers={**headers, "Content-Type": "application/json; charset=utf-8",
                                                                     "X-Requested-With": "XMLHttpRequest"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise IngestionError("Redirect without Location")
                current = validate_url(urljoin(current, location))
                continue
            if response.status == 429 or 500 <= response.status < 600:
                raise RetryableFetch(f"Source returned HTTP {response.status}; retry later")
            if response.status != 200:
                raise IngestionError(f"Source returned HTTP {response.status}")
            media_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type not in {"text/html", "application/xhtml+xml", "application/pdf", "application/json"}:
                raise IngestionError(f"Unsupported source content type: {media_type}")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise IngestionError("Compressed HTTP transfer rejected; identity encoding required")
            length = response.getheader("Content-Length")
            if length:
                try:
                    if int(length) > MAX_BYTES:
                        raise IngestionError("Source exceeds 12 MiB limit")
                except ValueError as exc:
                    raise IngestionError("Invalid Content-Length") from exc
            data = bytearray()
            while True:
                if time.monotonic() > deadline:
                    raise IngestionError("Source download exceeded time limit")
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise IngestionError("Source exceeds 12 MiB limit")
            if not data:
                raise IngestionError("Empty source response")
            return Download(original, current, bytes(data), media_type, datetime.now(timezone.utc),
                            {name: response.getheader(name) for name in ("ETag", "Last-Modified") if response.getheader(name)})
        finally:
            connection.close()
    raise IngestionError("Too many source redirects")


def fetch(url: str, body: bytes | None = None) -> Download:
    validate_url(url)
    for attempt in range(3):
        try:
            return _download(url, body)
        except ssl.SSLCertVerificationError as exc:
            raise IngestionError("Source TLS certificate verification failed") from exc
        except (OSError, http.client.HTTPException, RetryableFetch) as exc:
            if attempt == 2:
                raise IngestionError("Source fetch failed after three attempts") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")
