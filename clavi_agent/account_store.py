"""Persistent storage for local accounts backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type

from .account_constants import ROOT_ACCOUNT_ID
from .account_models import (
    AccountApiConfigRecord,
    AccountPasswordCredentialRecord,
    AccountRecord,
    AccountWebSessionRecord,
    AuthenticatedAccountSession,
    RootSeedResult,
)
from .llm_routing_models import LLMRoutingPolicy
from .sqlite_schema import configure_connection, ensure_agent_db_schema, utc_now_iso


DEFAULT_ROOT_USERNAME = "root"
DEFAULT_ROOT_DISPLAY_NAME = "Root"
DEFAULT_WEB_SESSION_TTL = timedelta(days=7)
PASSWORD_ALGO = "argon2id"


class AccountStore:
    """SQLite repository for persisted accounts and browser sessions."""

    _password_hasher = PasswordHasher(
        time_cost=2,
        memory_cost=12288,
        parallelism=1,
        hash_len=32,
        salt_len=16,
        type=Type.ID,
    )

    def __init__(
        self,
        db_path: str | Path,
        *,
        auto_seed_root: bool = True,
        root_username: str = DEFAULT_ROOT_USERNAME,
        root_display_name: str = DEFAULT_ROOT_DISPLAY_NAME,
        root_password: str | None = None,
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrap_root_password: str | None = None
        self._initialize()
        if auto_seed_root:
            seed_result = self.ensure_root_account(
                username=root_username,
                display_name=root_display_name,
                password=root_password,
            )
            self.bootstrap_root_password = seed_result.bootstrap_password

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self) -> None:
        with self._connect() as conn:
            ensure_agent_db_schema(conn)

    @staticmethod
    def _json_loads(raw: str | None, fallback: dict | None = None) -> dict:
        try:
            return json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return dict(fallback or {})

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = str(username or "").strip().lower()
        if not normalized:
            raise ValueError("Username is required.")
        return normalized

    @staticmethod
    def _normalize_display_name(display_name: str | None, *, fallback: str) -> str:
        normalized = str(display_name or "").strip()
        return normalized or fallback

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status or "").strip().lower() or "active"
        if normalized not in {"active", "disabled"}:
            raise ValueError(f"Unsupported account status: {status}")
        return normalized

    @staticmethod
    def _normalize_password(password: str) -> str:
        normalized = str(password or "")
        if len(normalized) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return normalized

    @staticmethod
    def _hash_session_token(session_token: str) -> str:
        normalized = str(session_token or "").strip()
        if not normalized:
            raise ValueError("Session token is required.")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _hash_password(cls, password: str) -> str:
        return cls._password_hasher.hash(cls._normalize_password(password))

    def _account_from_row(self, row: sqlite3.Row) -> AccountRecord:
        return AccountRecord(
            id=row["id"],
            username=row["username"],
            display_name=row["display_name"],
            status=row["status"],
            is_root=bool(row["is_root"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _password_credential_from_row(
        row: sqlite3.Row,
    ) -> AccountPasswordCredentialRecord:
        return AccountPasswordCredentialRecord(
            account_id=row["account_id"],
            password_hash=row["password_hash"],
            password_algo=row["password_algo"],
            password_updated_at=row["password_updated_at"],
        )

    @staticmethod
    def _web_session_from_row(row: sqlite3.Row) -> AccountWebSessionRecord:
        return AccountWebSessionRecord(
            id=row["id"],
            account_id=row["account_id"],
            session_token_hash=row["session_token_hash"],
            expires_at=row["expires_at"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            user_agent=row["user_agent"],
            ip=row["ip"],
        )

    @staticmethod
    def _api_config_from_row(row: sqlite3.Row) -> AccountApiConfigRecord:
        return AccountApiConfigRecord(
            id=row["id"],
            account_id=row["account_id"],
            name=row["name"],
            provider=row["provider"],
            api_base=row["api_base"],
            model=row["model"],
            api_key=row["api_key"],
            reasoning_enabled=bool(row["reasoning_enabled"]),
            llm_routing_policy=LLMRoutingPolicy.model_validate(
                AccountStore._json_loads(row["llm_routing_policy_json"])
            ),
            is_active=bool(row["is_active"]),
            last_used_at=row["last_used_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def generate_account_id() -> str:
        """Generate a new non-root account identifier."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_web_session_id() -> str:
        """Generate a new browser-session identifier."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_api_config_id() -> str:
        """Generate a new API configuration identifier."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_web_session_token() -> str:
        """Generate a random opaque browser-session token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def _generate_bootstrap_password() -> str:
        return secrets.token_urlsafe(18)

    @staticmethod
    def _normalize_api_config_name(name: str) -> str:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("Configuration name is required.")
        return normalized

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        normalized = str(provider or "anthropic").strip().lower() or "anthropic"
        if normalized not in {"anthropic", "openai"}:
            raise ValueError(f"Unsupported provider: {provider}")
        return normalized

    @staticmethod
    def _normalize_api_key(api_key: str) -> str:
        normalized = str(api_key or "").strip()
        if not normalized or normalized == "YOUR_API_KEY_HERE":
            raise ValueError("Please provide a valid API Key.")
        return normalized

    @staticmethod
    def _normalize_api_base(api_base: str | None) -> str:
        normalized = str(api_base or "").strip()
        return normalized or "https://api.minimax.io"

    @staticmethod
    def _normalize_model(model: str | None) -> str:
        normalized = str(model or "").strip()
        return normalized or "MiniMax-M2"

    @staticmethod
    def _normalize_llm_routing_policy(
        llm_routing_policy: LLMRoutingPolicy | dict | None,
    ) -> LLMRoutingPolicy:
        if isinstance(llm_routing_policy, LLMRoutingPolicy):
            return llm_routing_policy
        return LLMRoutingPolicy.model_validate(llm_routing_policy or {})

    def create_account(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
        account_id: str | None = None,
        status: str = "active",
        is_root: bool = False,
    ) -> dict:
        """Create one local account and its password credential."""
        normalized_username = self._normalize_username(username)
        normalized_display_name = self._normalize_display_name(
            display_name,
            fallback=normalized_username,
        )
        normalized_status = self._normalize_status(status)
        resolved_account_id = account_id or (
            ROOT_ACCOUNT_ID if is_root else self.generate_account_id()
        )
        password_hash = self._hash_password(password)
        now = utc_now_iso()

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO accounts (
                        id, username, display_name, status, is_root, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_account_id,
                        normalized_username,
                        normalized_display_name,
                        normalized_status,
                        int(is_root),
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO account_password_credentials (
                        account_id, password_hash, password_algo, password_updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        resolved_account_id,
                        password_hash,
                        PASSWORD_ALGO,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Account already exists for username or id: {normalized_username}"
            ) from exc

        account = self.get_account_record(resolved_account_id)
        if account is None:
            raise RuntimeError(f"Failed to create account: {resolved_account_id}")
        return account.model_dump(mode="python")

    def get_account_record(self, account_id: str) -> AccountRecord | None:
        """Get one account by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
        if not row:
            return None
        return self._account_from_row(row)

    def get_account(self, account_id: str) -> dict | None:
        """Get one account by id as a dict payload."""
        record = self.get_account_record(account_id)
        if record is None:
            return None
        return record.model_dump(mode="python")

    def get_account_by_username_record(self, username: str) -> AccountRecord | None:
        """Get one account by normalized username."""
        normalized_username = self._normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE username = ?",
                (normalized_username,),
            ).fetchone()
        if not row:
            return None
        return self._account_from_row(row)

    def get_account_by_username(self, username: str) -> dict | None:
        """Get one account by username as a dict payload."""
        record = self.get_account_by_username_record(username)
        if record is None:
            return None
        return record.model_dump(mode="python")

    def get_root_account_record(self) -> AccountRecord | None:
        """Get the persisted root account if it exists."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM accounts
                WHERE is_root = 1
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        return self._account_from_row(row)

    def list_account_records(self) -> list[AccountRecord]:
        """List all local accounts ordered with root first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM accounts
                ORDER BY is_root DESC, created_at ASC
                """
            ).fetchall()
        return [self._account_from_row(row) for row in rows]

    def list_accounts(self) -> list[dict]:
        """List all local accounts as dict payloads."""
        return [record.model_dump(mode="python") for record in self.list_account_records()]

    def update_account(
        self,
        account_id: str,
        *,
        display_name: str | None = None,
        status: str | None = None,
    ) -> dict | None:
        """Update mutable account fields."""
        record = self.get_account_record(account_id)
        if record is None:
            return None

        next_display_name = (
            self._normalize_display_name(display_name, fallback=record.username)
            if display_name is not None
            else record.display_name
        )
        next_status = (
            self._normalize_status(status)
            if status is not None
            else record.status
        )
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET display_name = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_display_name, next_status, now, account_id),
            )
        return self.get_account(account_id)

    def get_password_credential_record(
        self,
        account_id: str,
    ) -> AccountPasswordCredentialRecord | None:
        """Get the password credential for one account."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT account_id, password_hash, password_algo, password_updated_at
                FROM account_password_credentials
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()
        if not row:
            return None
        return self._password_credential_from_row(row)

    def get_password_credential(self, account_id: str) -> dict | None:
        """Get the password credential for one account as a dict payload."""
        record = self.get_password_credential_record(account_id)
        if record is None:
            return None
        return record.model_dump(mode="python")

    def set_password(self, account_id: str, password: str) -> dict:
        """Create or replace one account password credential."""
        account = self.get_account_record(account_id)
        if account is None:
            raise KeyError(f"Account not found: {account_id}")

        password_hash = self._hash_password(password)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO account_password_credentials (
                    account_id, password_hash, password_algo, password_updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    password_algo = excluded.password_algo,
                    password_updated_at = excluded.password_updated_at
                """,
                (account_id, password_hash, PASSWORD_ALGO, now),
            )
        credential = self.get_password_credential_record(account_id)
        if credential is None:
            raise RuntimeError(f"Failed to update password for account: {account_id}")
        return credential.model_dump(mode="python")

    def authenticate(self, username: str, password: str) -> AccountRecord | None:
        """Verify one username/password pair and return the active account."""
        account = self.get_account_by_username_record(username)
        if account is None or account.status != "active":
            return None

        credential = self.get_password_credential_record(account.id)
        if credential is None:
            return None

        try:
            verified = self._password_hasher.verify(
                credential.password_hash,
                password,
            )
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return None

        if not verified:
            return None

        if self._password_hasher.check_needs_rehash(credential.password_hash):
            self.set_password(account.id, password)

        return self.get_account_record(account.id)

    def create_web_session(
        self,
        account_id: str,
        *,
        session_token: str | None = None,
        expires_at: str | None = None,
        user_agent: str = "",
        ip: str = "",
    ) -> tuple[dict, str]:
        """Create one persisted browser session and return its raw token."""
        account = self.get_account_record(account_id)
        if account is None:
            raise KeyError(f"Account not found: {account_id}")
        if account.status != "active":
            raise ValueError("Disabled accounts cannot create browser sessions.")

        token = session_token or self.generate_web_session_token()
        token_hash = self._hash_session_token(token)
        now = utc_now_iso()
        resolved_expires_at = expires_at or (
            datetime.now(timezone.utc) + DEFAULT_WEB_SESSION_TTL
        ).isoformat(timespec="microseconds")
        session_id = self.generate_web_session_id()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO account_web_sessions (
                    id, account_id, session_token_hash, expires_at, created_at,
                    last_seen_at, user_agent, ip
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    account_id,
                    token_hash,
                    resolved_expires_at,
                    now,
                    now,
                    str(user_agent or "").strip(),
                    str(ip or "").strip(),
                ),
            )

        record = self.get_web_session_record(session_id)
        if record is None:
            raise RuntimeError(f"Failed to create browser session: {session_id}")
        return record.model_dump(mode="python"), token

    def get_web_session_record(self, session_id: str) -> AccountWebSessionRecord | None:
        """Get one browser session by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM account_web_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return self._web_session_from_row(row)

    def get_web_session(self, session_id: str) -> dict | None:
        """Get one browser session by id as a dict payload."""
        record = self.get_web_session_record(session_id)
        if record is None:
            return None
        return record.model_dump(mode="python")

    def get_authenticated_session(
        self,
        session_token: str,
        *,
        now: str | None = None,
    ) -> AuthenticatedAccountSession | None:
        """Resolve one active account plus its unexpired browser session."""
        token_hash = self._hash_session_token(session_token)
        effective_now = now or utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    s.id AS session_id,
                    s.account_id AS session_account_id,
                    s.session_token_hash,
                    s.expires_at,
                    s.created_at AS session_created_at,
                    s.last_seen_at,
                    s.user_agent,
                    s.ip,
                    a.id AS account_id,
                    a.username,
                    a.display_name,
                    a.status,
                    a.is_root,
                    a.created_at AS account_created_at,
                    a.updated_at AS account_updated_at
                FROM account_web_sessions AS s
                INNER JOIN accounts AS a
                    ON a.id = s.account_id
                WHERE s.session_token_hash = ?
                    AND s.expires_at > ?
                """,
                (token_hash, effective_now),
            ).fetchone()
        if not row:
            return None

        account = AccountRecord(
            id=row["account_id"],
            username=row["username"],
            display_name=row["display_name"],
            status=row["status"],
            is_root=bool(row["is_root"]),
            created_at=row["account_created_at"],
            updated_at=row["account_updated_at"],
        )
        if account.status != "active":
            return None

        web_session = AccountWebSessionRecord(
            id=row["session_id"],
            account_id=row["session_account_id"],
            session_token_hash=row["session_token_hash"],
            expires_at=row["expires_at"],
            created_at=row["session_created_at"],
            last_seen_at=row["last_seen_at"],
            user_agent=row["user_agent"],
            ip=row["ip"],
        )
        return AuthenticatedAccountSession(account=account, web_session=web_session)

    def touch_web_session(
        self,
        session_id: str,
        *,
        last_seen_at: str | None = None,
        expires_at: str | None = None,
    ) -> dict | None:
        """Refresh the last-seen timestamp for one browser session."""
        if self.get_web_session_record(session_id) is None:
            return None

        resolved_last_seen_at = last_seen_at or utc_now_iso()
        with self._connect() as conn:
            if expires_at is None:
                conn.execute(
                    """
                    UPDATE account_web_sessions
                    SET last_seen_at = ?
                    WHERE id = ?
                    """,
                    (resolved_last_seen_at, session_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE account_web_sessions
                    SET last_seen_at = ?, expires_at = ?
                    WHERE id = ?
                    """,
                    (resolved_last_seen_at, expires_at, session_id),
                )
        return self.get_web_session(session_id)

    def delete_web_session(self, session_id: str) -> bool:
        """Delete one browser session by id."""
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM account_web_sessions WHERE id = ?",
                (session_id,),
            ).rowcount
        return deleted > 0

    def delete_web_session_by_token(self, session_token: str) -> bool:
        """Delete one browser session by raw session token."""
        token_hash = self._hash_session_token(session_token)
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM account_web_sessions WHERE session_token_hash = ?",
                (token_hash,),
            ).rowcount
        return deleted > 0

    def delete_expired_web_sessions(self, *, now: str | None = None) -> int:
        """Delete expired browser sessions and return the affected row count."""
        effective_now = now or utc_now_iso()
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM account_web_sessions WHERE expires_at <= ?",
                (effective_now,),
            ).rowcount
        return deleted

    def get_api_config_record(
        self,
        config_id: str,
        *,
        account_id: str | None = None,
    ) -> AccountApiConfigRecord | None:
        """Get one persisted API configuration by id."""
        params: list[str] = [config_id]
        sql = "SELECT * FROM account_api_configs WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if not row:
            return None
        return self._api_config_from_row(row)

    def get_active_api_config_record(self, account_id: str) -> AccountApiConfigRecord | None:
        """Get the currently active API configuration for one account."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM account_api_configs
                WHERE account_id = ? AND is_active = 1
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
        if not row:
            return None
        return self._api_config_from_row(row)

    def list_api_config_records(self, account_id: str) -> list[AccountApiConfigRecord]:
        """List all stored API configurations for one account."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM account_api_configs
                WHERE account_id = ?
                ORDER BY is_active DESC, updated_at DESC, created_at DESC
                """,
                (account_id,),
            ).fetchall()
        return [self._api_config_from_row(row) for row in rows]

    def upsert_api_config(
        self,
        account_id: str,
        *,
        name: str,
        api_key: str,
        provider: str = "anthropic",
        api_base: str | None = None,
        model: str | None = None,
        reasoning_enabled: bool = False,
        llm_routing_policy: LLMRoutingPolicy | dict | None = None,
        activate: bool = True,
    ) -> AccountApiConfigRecord:
        """Create or update one named API configuration for an account."""
        account = self.get_account_record(account_id)
        if account is None:
            raise KeyError(f"Account not found: {account_id}")

        normalized_name = self._normalize_api_config_name(name)
        normalized_api_key = self._normalize_api_key(api_key)
        normalized_provider = self._normalize_provider(provider)
        normalized_api_base = self._normalize_api_base(api_base)
        normalized_model = self._normalize_model(model)
        normalized_llm_routing_policy = self._normalize_llm_routing_policy(llm_routing_policy)
        now = utc_now_iso()
        resolved_id = self.generate_api_config_id()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM account_api_configs
                WHERE account_id = ? AND name = ?
                """,
                (account_id, normalized_name),
            ).fetchone()
            if existing is not None:
                resolved_id = str(existing["id"])

            conn.execute(
                """
                INSERT INTO account_api_configs (
                    id, account_id, name, provider, api_base, model, api_key,
                    reasoning_enabled, llm_routing_policy_json, is_active, last_used_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, name) DO UPDATE SET
                    provider = excluded.provider,
                    api_base = excluded.api_base,
                    model = excluded.model,
                    api_key = excluded.api_key,
                    reasoning_enabled = excluded.reasoning_enabled,
                    llm_routing_policy_json = excluded.llm_routing_policy_json,
                    is_active = CASE
                        WHEN excluded.is_active = 1 THEN 1
                        ELSE account_api_configs.is_active
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    resolved_id,
                    account_id,
                    normalized_name,
                    normalized_provider,
                    normalized_api_base,
                    normalized_model,
                    normalized_api_key,
                    int(bool(reasoning_enabled)),
                    json.dumps(
                        normalized_llm_routing_policy.model_dump(mode="python"),
                        ensure_ascii=False,
                    ),
                    int(bool(activate)),
                    None,
                    now,
                    now,
                ),
            )
            if activate:
                conn.execute(
                    """
                    UPDATE account_api_configs
                    SET is_active = CASE WHEN id = ? THEN 1 ELSE 0 END
                    WHERE account_id = ?
                    """,
                    (resolved_id, account_id),
                )

        config = self.get_api_config_record(resolved_id, account_id=account_id)
        if config is None:
            raise RuntimeError(f"Failed to save API configuration: {resolved_id}")
        return config

    def activate_api_config(self, account_id: str, config_id: str) -> AccountApiConfigRecord:
        """Set one stored API configuration as the active one for an account."""
        record = self.get_api_config_record(config_id, account_id=account_id)
        if record is None:
            raise KeyError(f"API configuration not found: {config_id}")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE account_api_configs
                SET is_active = CASE WHEN id = ? THEN 1 ELSE 0 END
                WHERE account_id = ?
                """,
                (config_id, account_id),
            )
        refreshed = self.get_api_config_record(config_id, account_id=account_id)
        if refreshed is None:
            raise RuntimeError(f"Failed to activate API configuration: {config_id}")
        return refreshed

    def delete_api_config(self, account_id: str, config_id: str) -> bool:
        """Delete one stored API configuration."""
        with self._connect() as conn:
            deleted = conn.execute(
                """
                DELETE FROM account_api_configs
                WHERE id = ? AND account_id = ?
                """,
                (config_id, account_id),
            ).rowcount
        return deleted > 0

    def touch_api_config_last_used(
        self,
        account_id: str,
        config_id: str,
        *,
        used_at: str | None = None,
    ) -> AccountApiConfigRecord | None:
        """Update the last-used timestamp for one API configuration."""
        if self.get_api_config_record(config_id, account_id=account_id) is None:
            return None
        resolved_used_at = used_at or utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE account_api_configs
                SET last_used_at = ?
                WHERE id = ? AND account_id = ?
                """,
                (resolved_used_at, config_id, account_id),
            )
        return self.get_api_config_record(config_id, account_id=account_id)

    def ensure_root_account(
        self,
        *,
        username: str = DEFAULT_ROOT_USERNAME,
        display_name: str = DEFAULT_ROOT_DISPLAY_NAME,
        password: str | None = None,
    ) -> RootSeedResult:
        """Idempotently create the built-in root account and credential."""
        normalized_username = self._normalize_username(username)
        normalized_display_name = self._normalize_display_name(
            display_name,
            fallback=normalized_username,
        )
        root_account = self.get_root_account_record()
        created = False
        credential_created = False
        bootstrap_password: str | None = None

        if root_account is None:
            existing_account = self.get_account_by_username_record(normalized_username)
            if existing_account is not None:
                now = utc_now_iso()
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE accounts
                        SET is_root = 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (now, existing_account.id),
                    )
                root_account = self.get_account_record(existing_account.id)
            else:
                seed_password = password or self._generate_bootstrap_password()
                account_payload = self.create_account(
                    account_id=ROOT_ACCOUNT_ID,
                    username=normalized_username,
                    password=seed_password,
                    display_name=normalized_display_name,
                    status="active",
                    is_root=True,
                )
                root_account = AccountRecord.model_validate(account_payload)
                created = True
                credential_created = True
                if password is None:
                    bootstrap_password = seed_password
                return RootSeedResult(
                    account=root_account,
                    created=created,
                    credential_created=credential_created,
                    bootstrap_password=bootstrap_password,
                )

        if root_account is None:
            raise RuntimeError("Failed to resolve root account.")

        credential = self.get_password_credential_record(root_account.id)
        if credential is None:
            seed_password = password or self._generate_bootstrap_password()
            self.set_password(root_account.id, seed_password)
            credential_created = True
            if password is None:
                bootstrap_password = seed_password

        return RootSeedResult(
            account=root_account,
            created=created,
            credential_created=credential_created,
            bootstrap_password=bootstrap_password,
        )
