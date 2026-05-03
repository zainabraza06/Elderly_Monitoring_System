"""JWT + password hashing for API auth."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

SECRET = os.environ.get("JWT_SECRET", "sisfall-dev-change-me-in-production")
ALGO = "HS256"
TTL_DAYS = 7


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def create_token(*, user_id: str, role: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=TTL_DAYS)).timestamp()),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, SECRET, algorithms=[ALGO])
