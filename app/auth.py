"""Single static bearer token, checked against the API_SECRET env var - this is a
private endpoint for one app the user controls, not a multi-tenant service, so a
shared secret is sufficient (no user accounts/OAuth needed)."""

import os

from fastapi import Header, HTTPException

API_SECRET = os.environ.get("API_SECRET", "")


def require_api_key(authorization: str = Header(default="")) -> None:
    if not API_SECRET:
        # Fail closed rather than silently accepting every request if the
        # operator forgot to set the secret.
        raise HTTPException(status_code=500, detail="Server is not configured with API_SECRET")

    expected = f"Bearer {API_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
