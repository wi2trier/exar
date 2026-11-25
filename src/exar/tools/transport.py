"""What the tools that send requests share over HTTPX, which is imported only when they are built or called."""

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import anyio
import httpcore2
import httpx2

# The public alias is a stub when anyio is missing, which httpx2 requires, so the real class is imported.
from httpcore2._backends.anyio import AnyIOBackend
from pydantic import JsonValue

from . import Retry
from ._host import HostClient

__all__ = [
    "Response",
    "client_of",
    "media_type",
    "public_client",
    "read_capped",
    "request",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    """A response whose body was read up to a number of bytes.

    Args:
        url: Where the response came from, after the redirects of the client.
        status: The HTTP status code.
        headers: The response headers.
        encoding: The text encoding of the body, by its headers or HTTPX's guess.
        body: The bytes read.
        truncated: Whether the body went on past the bytes read.
    """

    url: str
    status: int
    headers: httpx2.Headers
    encoding: str | None
    body: bytes
    truncated: bool

    @property
    def is_error(self) -> bool:
        """Whether the status is a client or server error."""
        return self.status >= 400

    @property
    def text(self) -> str:
        """The body decoded by its encoding, UTF-8 if unknown, with undecodable bytes replaced."""
        return self.body.decode(self.encoding or "utf-8", errors="replace")


def _is_public(address: str) -> bool:
    """Whether *address* is globally routable, an IPv4-mapped IPv6 address judged by its IPv4 address.

    Examples:
        >>> [_is_public(a) for a in ("93.184.215.14", "127.0.0.1", "169.254.169.254", "::ffff:10.0.0.1", "fe80::1%eth0")]
        [True, False, False, False, False]
    """
    ip = ipaddress.ip_address(address)

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    return ip.is_global and not ip.is_multicast


class _PublicBackend(AnyIOBackend):
    """Connects to public addresses alone, checked on the address connected to, so neither a redirect nor DNS rebinding evades it."""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        try:
            with anyio.fail_after(timeout):
                infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except TimeoutError as error:
            raise httpcore2.ConnectTimeout(f"resolving {host} timed out") from error
        except OSError as error:
            raise httpcore2.ConnectError(f"cannot resolve {host}: {error}") from error

        public = [str(info[4][0]) for info in infos if _is_public(str(info[4][0]))]

        if not public:
            raise httpcore2.ConnectError(f"{host} resolves to no public address")

        return await super().connect_tcp(
            public[0], port, timeout, local_address, socket_options
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        raise httpcore2.ConnectError("unix sockets are not public")


def public_client(
    *, max_redirects: int = 5, headers: Mapping[str, str] | None = None
) -> httpx2.AsyncClient:
    """A client that reaches public addresses alone, for tools such as `WebFetch` whose URLs come from untrusted pages.

    Every connection, that of each redirect hop included, is refused unless the host resolves to a globally routable address,
    which is the one connected to.
    Proxy variables are ignored, since a proxy would connect on the client's behalf unchecked.

    Args:
        max_redirects: Redirect hops followed.
        headers: Headers sent with every request.
    """
    transport = httpx2.AsyncHTTPTransport(trust_env=False)
    # HTTPX takes no network backend, so its pool is replaced by one with the checking backend.
    transport._pool = httpcore2.AsyncConnectionPool(
        ssl_context=httpx2.create_ssl_context(), network_backend=_PublicBackend()
    )

    return httpx2.AsyncClient(
        transport=transport,
        trust_env=False,
        follow_redirects=True,
        max_redirects=max_redirects,
        headers=headers,
    )


async def read_capped(response: httpx2.Response, max_bytes: int) -> tuple[bytes, bool]:
    """Streams the response body up to a number of bytes.

    HTTPX decompresses in pieces of at most a mebibyte, so a compression bomb overshoots the cap by no more than one piece.
    """
    buffer = bytearray()

    async for chunk in response.aiter_bytes():
        buffer += chunk

        if len(buffer) > max_bytes:
            del buffer[max_bytes:]

            return bytes(buffer), True

    return bytes(buffer), False


def client_of(host: HostClient) -> httpx2.AsyncClient:
    """The client the host bound.

    Raises:
        exar.errors.Unbound: If no function is bound to the name.
        TypeError: If the bound function returns something else than an `httpx2.AsyncClient`.
    """
    client = host()

    if not isinstance(client, httpx2.AsyncClient):
        raise TypeError(f"Client {host.name!r} is not an httpx2.AsyncClient")

    return client


def media_type(headers: httpx2.Headers, /) -> str:
    """The bare lowercase media type of the Content-Type header, empty without one.

    Examples:
        >>> media_type(httpx2.Headers({"Content-Type": "Text/HTML; charset=utf-8"}))
        'text/html'
    """
    return headers.get("content-type", "").partition(";")[0].strip().lower()


async def request(
    client: HostClient,
    method: str,
    url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    params: Mapping[str, str | list[str]] | None = None,
    headers: Mapping[str, str] | None = None,
    json: JsonValue = None,
    check: Callable[[httpx2.Response], None] | None = None,
) -> Response:
    """Sends a request through the client of the host and reads the response body up to `max_bytes`.

    `check` sees the response before its body is read, so it can refuse one, such as by its content type, without downloading it.

    Args:
        client: The client the host bound.
        method: The HTTP method.
        url: Where to send it.
        timeout_seconds: Bound of the request as a whole, the body included.
        max_bytes: Bytes of the body that are read.
        params: The query parameters.
        headers: The headers besides those of the client.
        json: The JSON body, none if None.
        check: Raises `Retry` for a response not worth reading, none if None.

    Raises:
        Retry: If the request timed out or failed, such as by the URL policy of the client.
        exar.errors.Unbound: If no function is bound to the name of the client.
        TypeError: If the bound function returns something else than an `httpx2.AsyncClient`.
    """
    try:
        async with (
            asyncio.timeout(timeout_seconds),
            client_of(client).stream(
                method, url, params=params, headers=headers, json=json
            ) as response,
        ):
            if check is not None:
                check(response)

            body, truncated = await read_capped(response, max_bytes)
    except (TimeoutError, httpx2.TimeoutException) as error:
        raise Retry("request timed out.") from error
    except httpx2.RequestError as error:
        # The client refuses a URL its policy denies by raising one too.
        raise Retry(str(error) or "request failed.") from error

    return Response(
        url=str(response.url),
        status=response.status_code,
        headers=response.headers,
        encoding=response.encoding,
        body=body,
        truncated=truncated,
    )
