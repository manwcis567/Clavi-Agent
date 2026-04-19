"""Typed models for persisted accounts, credentials, and web sessions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm_routing_models import LLMRoutingPolicy


AccountStatus = Literal["active", "disabled"]
PasswordAlgorithm = Literal["argon2id"]


class AccountRecord(BaseModel):
    """Persisted local account metadata."""

    id: str
    username: str
    display_name: str
    status: AccountStatus = "active"
    is_root: bool = False
    created_at: str
    updated_at: str


class AccountPasswordCredentialRecord(BaseModel):
    """Persisted password credential for one account."""

    account_id: str
    password_hash: str
    password_algo: PasswordAlgorithm = "argon2id"
    password_updated_at: str


class AccountWebSessionRecord(BaseModel):
    """Persisted browser login session."""

    id: str
    account_id: str
    session_token_hash: str
    expires_at: str
    created_at: str
    last_seen_at: str
    user_agent: str = ""
    ip: str = ""


class AccountApiConfigRecord(BaseModel):
    """Persisted per-account LLM/API credential set."""

    id: str
    account_id: str
    name: str
    provider: Literal["anthropic", "openai"] = "anthropic"
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2"
    api_key: str
    reasoning_enabled: bool = False
    llm_routing_policy: LLMRoutingPolicy = Field(default_factory=LLMRoutingPolicy)
    is_active: bool = False
    last_used_at: str | None = None
    created_at: str
    updated_at: str


class AuthenticatedAccountSession(BaseModel):
    """Joined view for one validated browser session."""

    account: AccountRecord
    web_session: AccountWebSessionRecord


class RootSeedResult(BaseModel):
    """Result payload for idempotent root-account seeding."""

    account: AccountRecord
    created: bool = False
    credential_created: bool = False
    bootstrap_password: str | None = None
