# SPDX-License-Identifier: Apache-2.0
"""URL validation helpers for server-side fetches."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def validate_public_http_url(url: str, *, field_name: str = "URL") -> str:
    """Validate an http(s) URL is not obviously private/internal."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) {field_name} values are supported.")
    if not parsed.hostname:
        raise ValueError(f"{field_name} must include a host.")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(
            f"Could not resolve {field_name} host: {parsed.hostname}"
        ) from exc

    for addr in addresses:
        ip_text = addr[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} host address: {ip_text}") from exc
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"{field_name} host resolves to a non-public address.")

    return url
