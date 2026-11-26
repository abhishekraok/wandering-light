"""HTTP protections shared by the two local policy debuggers."""

from __future__ import annotations

from ipaddress import ip_address
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

if TYPE_CHECKING:
    from fastapi import Request, Response
    from starlette.middleware.base import RequestResponseEndpoint


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


async def local_debugger_requests(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    try:
        host = request.url.hostname
    except ValueError:
        host = None
    if host is None or not is_loopback_host(host):
        return PlainTextResponse(
            "The debugger requires a loopback host.", status_code=400
        )
    if request.url.path.startswith("/api/") and request.headers.get(
        "sec-fetch-site"
    ) in {"same-site", "cross-site"}:
        return PlainTextResponse("Cross-origin debugger request.", status_code=403)
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    return response


def parse_op_path(path: str) -> list[str]:
    steps = [name for name in path.split(",") if name]
    if len(steps) > 128:
        raise HTTPException(
            status_code=422, detail="A debugger path can have up to 128 ops."
        )
    return steps
