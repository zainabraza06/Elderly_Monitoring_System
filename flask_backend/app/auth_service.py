from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from .schemas import (
    AuthLoginRequest,
    AuthSessionResponse,
    AuthSignupPatientRequest,
    AuthUserProfile,
    UserRole,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AuthContext:
    user_id: str
    email: str
    selected_role: UserRole
    patient_id: str | None
    available_roles: list[UserRole]


class AuthService:
    def __init__(
        self,
        *,
        token_ttl_seconds: int = 60 * 60 * 24,
        secret: str | None = None,
        mongo_uri: str | None = None,
        mongo_database: str | None = None,
    ) -> None:
        resolved_secret = (secret or os.getenv("EMS_AUTH_SECRET", "")).strip()
        if not resolved_secret:
            resolved_secret = secrets.token_hex(32)
            logger.warning(
                "EMS_AUTH_SECRET is not configured. Using an ephemeral auth secret for this process."
            )

        self._secret = resolved_secret.encode("utf-8")
        self._token_ttl_seconds = max(60, int(token_ttl_seconds))

        self._mongo_uri = (mongo_uri or os.getenv("EMS_MONGO_URI", "mongodb://localhost:27017")).strip()
        if not self._mongo_uri:
            self._mongo_uri = "mongodb://localhost:27017"

        self._mongo_database_name = (
            mongo_database or os.getenv("EMS_MONGO_DATABASE", "elderly_monitoring")
        ).strip()
        if not self._mongo_database_name:
            self._mongo_database_name = "elderly_monitoring"

        self._users: dict[str, dict[str, Any]] = {}
        self._email_index: dict[str, str] = {}
        self._role_grants: dict[tuple[str, str], str | None] = {}

        self._mongo_client: AsyncIOMotorClient | None = None
        self._mongo_db = None
        self._users_collection = None
        self._role_grants_collection = None
        self._mongo_ready = False
        self._mongo_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "backend": "mongodb" if self._mongo_ready else "memory",
            "enabled": self._mongo_ready,
            "error": self._mongo_error,
            "uri": self._mongo_uri,
            "database": self._mongo_database_name,
            "users_in_memory": len(self._users),
        }

    async def startup(self) -> None:
        try:
            self._mongo_client = AsyncIOMotorClient(
                self._mongo_uri,
                serverSelectionTimeoutMS=3000,
            )
            await self._mongo_client.admin.command("ping")

            self._mongo_db = self._mongo_client[self._mongo_database_name]
            self._users_collection = self._mongo_db["auth_users"]
            self._role_grants_collection = self._mongo_db["auth_role_grants"]

            await self._users_collection.create_index("email", unique=True)
            await self._users_collection.create_index("id", unique=True)
            await self._role_grants_collection.create_index(
                [("user_id", ASCENDING), ("role", ASCENDING)],
                unique=True,
            )

            self._mongo_ready = True
            self._mongo_error = None
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._mongo_error = (
                "MongoDB auth persistence is unavailable. Falling back to in-memory mode. "
                f"Details: {exc}"
            )
            logger.warning(self._mongo_error)
            self._mongo_ready = False

            if self._mongo_client is not None:
                self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            self._users_collection = None
            self._role_grants_collection = None

    async def shutdown(self) -> None:
        if self._mongo_client is not None:
            self._mongo_client.close()

    async def ensure_email_available(self, email: str) -> None:
        normalized = self._normalize_email(email)
        existing = await self._load_user_by_email(normalized)
        if existing is not None:
            raise ValueError("An account with this email already exists.")

    async def register_patient_user(
        self,
        payload: AuthSignupPatientRequest,
        *,
        patient_id: str,
    ) -> AuthSessionResponse:
        await self.ensure_email_available(payload.email)

        user_id = f"usr_{uuid4().hex[:12]}"
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(payload.password, salt)
        now = datetime.now(timezone.utc)

        user_record = {
            "id": user_id,
            "email": self._normalize_email(payload.email),
            "display_name": payload.full_name.strip(),
            "password_hash": password_hash,
            "password_salt": salt,
            "created_at": now,
        }

        await self._create_user(user_record)
        await self._upsert_role_grant(user_id=user_id, role=UserRole.patient, patient_id=patient_id)

        profile = await self.get_user_profile(user_id)
        return self._build_session(profile=profile, selected_role=UserRole.patient)

    async def login(self, payload: AuthLoginRequest) -> AuthSessionResponse:
        normalized_email = self._normalize_email(payload.email)
        user = await self._load_user_by_email(normalized_email)
        if user is None:
            raise ValueError("Invalid email or password.")

        expected_hash = self._hash_password(payload.password, str(user["password_salt"]))
        if not hmac.compare_digest(expected_hash, str(user["password_hash"])):
            raise ValueError("Invalid email or password.")

        profile = await self.get_user_profile(str(user["id"]))
        if payload.role not in profile.available_roles:
            raise ValueError("This account does not have access to the selected role.")

        return self._build_session(profile=profile, selected_role=payload.role)

    async def switch_role(self, *, user_id: str, role: UserRole) -> AuthSessionResponse:
        profile = await self.get_user_profile(user_id)
        if role not in profile.available_roles:
            raise ValueError("This account does not have access to the selected role.")
        return self._build_session(profile=profile, selected_role=role)

    async def enable_caregiver_role(self, *, user_id: str) -> AuthUserProfile:
        grants = await self._list_role_grants(user_id)
        patient_grant = next((grant for grant in grants if grant["role"] == UserRole.patient.value), None)
        if patient_grant is None or patient_grant.get("patient_id") is None:
            raise ValueError("A patient role must exist before enabling caregiver role.")

        await self._upsert_role_grant(
            user_id=user_id,
            role=UserRole.caregiver,
            patient_id=str(patient_grant.get("patient_id")),
        )
        return await self.get_user_profile(user_id)

    async def authenticate_header(self, authorization_header: str | None) -> AuthContext:
        if authorization_header is None or not authorization_header.strip():
            raise ValueError("Missing Authorization header.")

        prefix = "Bearer "
        if not authorization_header.startswith(prefix):
            raise ValueError("Authorization header must use Bearer token format.")

        token = authorization_header[len(prefix):].strip()
        if not token:
            raise ValueError("Access token is missing.")

        payload = self._verify_token(token)
        user_id = str(payload.get("uid", ""))
        role_raw = str(payload.get("role", ""))

        if not user_id or not role_raw:
            raise ValueError("Invalid access token payload.")

        try:
            selected_role = UserRole(role_raw)
        except ValueError as exc:
            raise ValueError("Invalid access token role.") from exc

        profile = await self.get_user_profile(user_id)
        if selected_role not in profile.available_roles:
            raise ValueError("This access token role is no longer valid.")

        grant_map = await self._role_map(user_id)
        patient_id = grant_map.get(selected_role)

        return AuthContext(
            user_id=profile.user_id,
            email=profile.email,
            selected_role=selected_role,
            patient_id=patient_id,
            available_roles=profile.available_roles,
        )

    async def get_user_profile(self, user_id: str) -> AuthUserProfile:
        user = await self._load_user_by_id(user_id)
        if user is None:
            raise ValueError("Account was not found.")

        role_map = await self._role_map(user_id)
        available_roles = sorted(role_map.keys(), key=lambda item: item.value)
        patient_id = role_map.get(UserRole.patient) or role_map.get(UserRole.caregiver)

        return AuthUserProfile(
            user_id=str(user["id"]),
            email=self._normalize_email(str(user["email"])),
            display_name=str(user.get("display_name") or "User"),
            available_roles=available_roles,
            patient_id=patient_id,
        )

    def _build_session(self, *, profile: AuthUserProfile, selected_role: UserRole) -> AuthSessionResponse:
        role_patient_id = profile.patient_id

        token = self._issue_token(
            user_id=profile.user_id,
            email=profile.email,
            role=selected_role,
            patient_id=role_patient_id,
        )
        return AuthSessionResponse(
            access_token=token,
            token_type="bearer",
            selected_role=selected_role,
            user=profile,
        )

    async def _load_user_by_email(self, normalized_email: str) -> dict[str, Any] | None:
        if self._mongo_ready and self._users_collection is not None:
            row = await self._users_collection.find_one({"email": normalized_email})
            return self._user_row_to_dict(row)

        user_id = self._email_index.get(normalized_email)
        if user_id is None:
            return None
        return self._users.get(user_id)

    async def _load_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        if self._mongo_ready and self._users_collection is not None:
            row = await self._users_collection.find_one({"id": user_id})
            return self._user_row_to_dict(row)

        return self._users.get(user_id)

    async def _create_user(self, record: dict[str, Any]) -> None:
        if self._mongo_ready and self._users_collection is not None:
            try:
                await self._users_collection.insert_one(record)
            except DuplicateKeyError as exc:
                raise ValueError("An account with this email already exists.") from exc
            return

        self._users[str(record["id"])] = record
        self._email_index[self._normalize_email(str(record["email"]))] = str(record["id"])

    async def _upsert_role_grant(self, *, user_id: str, role: UserRole, patient_id: str | None) -> None:
        if self._mongo_ready and self._role_grants_collection is not None:
            await self._role_grants_collection.update_one(
                {"user_id": user_id, "role": role.value},
                {
                    "$set": {"patient_id": patient_id},
                    "$setOnInsert": {
                        "user_id": user_id,
                        "role": role.value,
                        "created_at": datetime.now(timezone.utc),
                    },
                },
                upsert=True,
            )
            return

        self._role_grants[(user_id, role.value)] = patient_id

    async def _list_role_grants(self, user_id: str) -> list[dict[str, Any]]:
        if self._mongo_ready and self._role_grants_collection is not None:
            rows = await self._role_grants_collection.find({"user_id": user_id}).to_list(length=None)
            grants: list[dict[str, Any]] = []
            for row in rows:
                grants.append(
                    {
                        "user_id": str(row.get("user_id", "")),
                        "role": str(row.get("role", "")),
                        "patient_id": row.get("patient_id"),
                    }
                )
            return grants

        grants = []
        for (grant_user_id, role), patient_id in self._role_grants.items():
            if grant_user_id != user_id:
                continue
            grants.append(
                {
                    "user_id": user_id,
                    "role": role,
                    "patient_id": patient_id,
                }
            )
        return grants

    async def _role_map(self, user_id: str) -> dict[UserRole, str | None]:
        grants = await self._list_role_grants(user_id)
        role_map: dict[UserRole, str | None] = {}
        for grant in grants:
            try:
                role = UserRole(str(grant.get("role")))
            except ValueError:
                continue
            role_map[role] = grant.get("patient_id")
        return role_map

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _hash_password(password: str, salt_hex: str) -> str:
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return digest.hex()

    def _issue_token(
        self,
        *,
        user_id: str,
        email: str,
        role: UserRole,
        patient_id: str | None,
    ) -> str:
        now_ts = int(time.time())
        payload = {
            "uid": user_id,
            "email": email,
            "role": role.value,
            "patient_id": patient_id,
            "iat": now_ts,
            "exp": now_ts + self._token_ttl_seconds,
        }
        return self._encode_signed_payload(payload)

    def _verify_token(self, token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid access token format.")

        header_part, payload_part, signature_part = parts
        signed_data = f"{header_part}.{payload_part}".encode("utf-8")
        expected_signature = hmac.new(self._secret, signed_data, hashlib.sha256).digest()
        provided_signature = self._b64url_decode(signature_part)

        if not hmac.compare_digest(expected_signature, provided_signature):
            raise ValueError("Invalid access token signature.")

        payload_bytes = self._b64url_decode(payload_part)
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Invalid access token payload.") from exc

        exp = int(payload.get("exp", 0))
        if int(time.time()) >= exp:
            raise ValueError("Access token has expired.")

        return payload

    def _encode_signed_payload(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_part = self._b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_part = self._b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signed_data = f"{header_part}.{payload_part}".encode("utf-8")
        signature = hmac.new(self._secret, signed_data, hashlib.sha256).digest()
        signature_part = self._b64url_encode(signature)
        return f"{header_part}.{payload_part}.{signature_part}"

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    @staticmethod
    def _b64url_decode(value: str) -> bytes:
        padding = "=" * ((4 - len(value) % 4) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("utf-8"))

    @staticmethod
    def _user_row_to_dict(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None

        cleaned = dict(row)
        cleaned.pop("_id", None)
        return {
            "id": cleaned.get("id", ""),
            "email": cleaned.get("email", ""),
            "display_name": cleaned.get("display_name", "User"),
            "password_hash": cleaned.get("password_hash", ""),
            "password_salt": cleaned.get("password_salt", ""),
            "created_at": cleaned.get("created_at", datetime.now(timezone.utc)),
        }
