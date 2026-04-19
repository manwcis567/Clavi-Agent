"""消息渠道集成的 SQLite 仓储实现。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .integration_models import (
    DEFAULT_MAX_INBOUND_HEADERS_BYTES,
    DEFAULT_MAX_INBOUND_PAYLOAD_BYTES,
    ConversationBindingRecord,
    DeliveryAttemptRecord,
    InboundEventRecord,
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
    OutboundDeliveryRecord,
    RoutingRuleRecord,
    prepare_bounded_json_payload,
)
from .sqlite_schema import configure_connection, ensure_session_db_schema


class _IntegrationStoreBase:
    """集成仓储的公共基础能力。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self) -> None:
        with self._connect() as conn:
            ensure_session_db_schema(conn)

    @staticmethod
    def _json_dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_load(payload: str | None, fallback: Any) -> Any:
        if not payload:
            return fallback
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _resolve_account_id(
        conn: sqlite3.Connection,
        table_name: str,
        id_column: str,
        id_value: str,
        *,
        fallback: str = ROOT_ACCOUNT_ID,
    ) -> str:
        row = conn.execute(
            f"SELECT account_id FROM {table_name} WHERE {id_column} = ?",
            (id_value,),
        ).fetchone()
        if row is None:
            return fallback
        return str(row["account_id"] or fallback)


class IntegrationStore(_IntegrationStoreBase):
    """管理集成配置、凭证与路由规则。"""

    def _integration_from_row(self, row: sqlite3.Row) -> IntegrationConfigRecord:
        return IntegrationConfigRecord(
            id=row["id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            name=row["name"],
            kind=row["kind"],
            status=row["status"],
            display_name=row["display_name"],
            tenant_id=row["tenant_id"],
            webhook_path=row["webhook_path"],
            config=self._json_load(row["config_json"], {}),
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_verified_at=row["last_verified_at"],
            last_error=row["last_error"],
        )

    def _credential_from_row(self, row: sqlite3.Row) -> IntegrationCredentialRecord:
        return IntegrationCredentialRecord(
            id=row["id"],
            integration_id=row["integration_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            credential_key=row["credential_key"],
            storage_kind=row["storage_kind"],
            secret_ref=row["secret_ref"],
            secret_ciphertext=row["secret_ciphertext"],
            masked_value=row["masked_value"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _routing_rule_from_row(self, row: sqlite3.Row) -> RoutingRuleRecord:
        return RoutingRuleRecord(
            id=row["id"],
            integration_id=row["integration_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            priority=row["priority"],
            match_type=row["match_type"],
            match_value=row["match_value"],
            agent_id=row["agent_id"],
            session_strategy=row["session_strategy"],
            enabled=bool(row["enabled"]),
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_integration(self, record: IntegrationConfigRecord) -> IntegrationConfigRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO integrations (
                    id, account_id, name, kind, status, display_name, tenant_id, webhook_path,
                    config_json, metadata_json, created_at, updated_at, last_verified_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.account_id,
                    record.name,
                    record.kind,
                    record.status,
                    record.display_name,
                    record.tenant_id,
                    record.webhook_path,
                    self._json_dump(record.config),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.last_verified_at,
                    record.last_error,
                ),
            )
        return record

    def get_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationConfigRecord | None:
        params: list[Any] = [integration_id]
        sql = "SELECT * FROM integrations WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._integration_from_row(row)

    def list_integrations(
        self,
        *,
        account_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[IntegrationConfigRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM integrations"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._integration_from_row(row) for row in rows]

    def update_integration(self, record: IntegrationConfigRecord) -> IntegrationConfigRecord:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE integrations
                SET account_id = ?, name = ?, kind = ?, status = ?, display_name = ?, tenant_id = ?, webhook_path = ?,
                    config_json = ?, metadata_json = ?, created_at = ?, updated_at = ?,
                    last_verified_at = ?, last_error = ?
                WHERE id = ?
                """,
                (
                    record.account_id,
                    record.name,
                    record.kind,
                    record.status,
                    record.display_name,
                    record.tenant_id,
                    record.webhook_path,
                    self._json_dump(record.config),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.last_verified_at,
                    record.last_error,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Integration not found: {record.id}")
        return record

    def create_credential(
        self, record: IntegrationCredentialRecord
    ) -> IntegrationCredentialRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            conn.execute(
                """
                INSERT INTO integration_credentials (
                    id, integration_id, account_id, credential_key, storage_kind, secret_ref,
                    secret_ciphertext, masked_value, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.integration_id,
                    resolved_account_id,
                    record.credential_key,
                    record.storage_kind,
                    record.secret_ref,
                    record.secret_ciphertext,
                    record.masked_value,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def get_credential(
        self,
        credential_id: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationCredentialRecord | None:
        params: list[Any] = [credential_id]
        sql = "SELECT * FROM integration_credentials WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._credential_from_row(row)

    def get_credential_by_key(
        self,
        integration_id: str,
        credential_key: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationCredentialRecord | None:
        params: list[Any] = [integration_id, credential_key]
        sql = """
            SELECT *
            FROM integration_credentials
            WHERE integration_id = ? AND credential_key = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._credential_from_row(row)

    def list_credentials(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> list[IntegrationCredentialRecord]:
        params: list[Any] = [integration_id]
        sql = """
            SELECT *
            FROM integration_credentials
            WHERE integration_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY credential_key ASC, updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                tuple(params),
            ).fetchall()
        return [self._credential_from_row(row) for row in rows]

    def update_credential(
        self, record: IntegrationCredentialRecord
    ) -> IntegrationCredentialRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            updated = conn.execute(
                """
                UPDATE integration_credentials
                SET integration_id = ?, account_id = ?, credential_key = ?, storage_kind = ?, secret_ref = ?,
                    secret_ciphertext = ?, masked_value = ?, metadata_json = ?, created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.integration_id,
                    resolved_account_id,
                    record.credential_key,
                    record.storage_kind,
                    record.secret_ref,
                    record.secret_ciphertext,
                    record.masked_value,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Integration credential not found: {record.id}")
        return record.model_copy(update={"account_id": resolved_account_id})

    def delete_credential(self, credential_id: str, *, account_id: str | None = None) -> bool:
        params: list[Any] = [credential_id]
        sql = "DELETE FROM integration_credentials WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0

    def create_routing_rule(self, record: RoutingRuleRecord) -> RoutingRuleRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            conn.execute(
                """
                INSERT INTO routing_rules (
                    id, integration_id, account_id, priority, match_type, match_value, agent_id,
                    session_strategy, enabled, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.integration_id,
                    resolved_account_id,
                    record.priority,
                    record.match_type,
                    record.match_value,
                    record.agent_id,
                    record.session_strategy,
                    int(record.enabled),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def get_routing_rule(
        self,
        rule_id: str,
        *,
        account_id: str | None = None,
    ) -> RoutingRuleRecord | None:
        params: list[Any] = [rule_id]
        sql = "SELECT * FROM routing_rules WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._routing_rule_from_row(row)

    def list_routing_rules(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[RoutingRuleRecord]:
        params: list[Any] = [integration_id]
        sql = """
            SELECT *
            FROM routing_rules
            WHERE integration_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        if enabled is not None:
            sql += " AND enabled = ?"
            params.append(int(enabled))
        sql += " ORDER BY priority ASC, updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._routing_rule_from_row(row) for row in rows]

    def update_routing_rule(self, record: RoutingRuleRecord) -> RoutingRuleRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            updated = conn.execute(
                """
                UPDATE routing_rules
                SET integration_id = ?, account_id = ?, priority = ?, match_type = ?, match_value = ?, agent_id = ?,
                    session_strategy = ?, enabled = ?, metadata_json = ?, created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.integration_id,
                    resolved_account_id,
                    record.priority,
                    record.match_type,
                    record.match_value,
                    record.agent_id,
                    record.session_strategy,
                    int(record.enabled),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Routing rule not found: {record.id}")
        return record.model_copy(update={"account_id": resolved_account_id})

    def delete_routing_rule(self, rule_id: str, *, account_id: str | None = None) -> bool:
        params: list[Any] = [rule_id]
        sql = "DELETE FROM routing_rules WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0


class InboundEventStore(_IntegrationStoreBase):
    """管理渠道入站事件。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_headers_bytes: int = DEFAULT_MAX_INBOUND_HEADERS_BYTES,
        max_payload_bytes: int = DEFAULT_MAX_INBOUND_PAYLOAD_BYTES,
    ):
        self.max_headers_bytes = max_headers_bytes
        self.max_payload_bytes = max_payload_bytes
        super().__init__(db_path)

    def _event_from_row(self, row: sqlite3.Row) -> InboundEventRecord:
        return InboundEventRecord(
            id=row["id"],
            integration_id=row["integration_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            provider_event_id=row["provider_event_id"] or "",
            provider_message_id=row["provider_message_id"] or "",
            provider_chat_id=row["provider_chat_id"],
            provider_thread_id=row["provider_thread_id"],
            provider_user_id=row["provider_user_id"],
            event_type=row["event_type"],
            received_at=row["received_at"],
            signature_valid=bool(row["signature_valid"]),
            dedup_key=row["dedup_key"],
            raw_headers=self._json_load(row["raw_headers_json"], {}),
            raw_headers_size_bytes=row["raw_headers_size_bytes"],
            raw_headers_truncated=bool(row["raw_headers_truncated"]),
            raw_headers_redacted_fields=self._json_load(
                row["raw_headers_redacted_fields_json"],
                [],
            ),
            raw_payload=self._json_load(row["raw_payload_json"], {}),
            raw_payload_size_bytes=row["raw_payload_size_bytes"],
            raw_payload_truncated=bool(row["raw_payload_truncated"]),
            raw_payload_redacted_fields=self._json_load(
                row["raw_payload_redacted_fields_json"],
                [],
            ),
            normalized_status=row["normalized_status"],
            normalized_error=row["normalized_error"],
            metadata=self._json_load(row["metadata_json"], {}),
        )

    def _sanitize_event(self, record: InboundEventRecord) -> InboundEventRecord:
        headers_payload = prepare_bounded_json_payload(
            record.raw_headers,
            max_bytes=self.max_headers_bytes,
        )
        body_payload = prepare_bounded_json_payload(
            record.raw_payload,
            max_bytes=self.max_payload_bytes,
        )
        return record.model_copy(
            update={
                "raw_headers": headers_payload.data,
                "raw_headers_size_bytes": headers_payload.size_bytes,
                "raw_headers_truncated": headers_payload.truncated,
                "raw_headers_redacted_fields": headers_payload.redacted_fields,
                "raw_payload": body_payload.data,
                "raw_payload_size_bytes": body_payload.size_bytes,
                "raw_payload_truncated": body_payload.truncated,
                "raw_payload_redacted_fields": body_payload.redacted_fields,
            }
        )

    def create_event(self, record: InboundEventRecord) -> InboundEventRecord:
        sanitized = self._sanitize_event(record)
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                sanitized.integration_id,
                fallback=sanitized.account_id,
            )
            conn.execute(
                """
                INSERT INTO inbound_events (
                    id, integration_id, account_id, provider_event_id, provider_message_id, provider_chat_id,
                    provider_thread_id, provider_user_id, event_type, received_at, signature_valid,
                    dedup_key, raw_headers_json, raw_headers_size_bytes, raw_headers_truncated,
                    raw_headers_redacted_fields_json, raw_payload_json, raw_payload_size_bytes,
                    raw_payload_truncated, raw_payload_redacted_fields_json, normalized_status,
                    normalized_error, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sanitized.id,
                    sanitized.integration_id,
                    resolved_account_id,
                    sanitized.provider_event_id or None,
                    sanitized.provider_message_id or None,
                    sanitized.provider_chat_id,
                    sanitized.provider_thread_id,
                    sanitized.provider_user_id,
                    sanitized.event_type,
                    sanitized.received_at,
                    int(sanitized.signature_valid),
                    sanitized.dedup_key,
                    self._json_dump(sanitized.raw_headers),
                    sanitized.raw_headers_size_bytes,
                    int(sanitized.raw_headers_truncated),
                    self._json_dump(sanitized.raw_headers_redacted_fields),
                    self._json_dump(sanitized.raw_payload),
                    sanitized.raw_payload_size_bytes,
                    int(sanitized.raw_payload_truncated),
                    self._json_dump(sanitized.raw_payload_redacted_fields),
                    sanitized.normalized_status,
                    sanitized.normalized_error,
                    self._json_dump(sanitized.metadata),
                ),
            )
        return sanitized.model_copy(update={"account_id": resolved_account_id})

    def get_event(
        self,
        event_id: str,
        *,
        account_id: str | None = None,
    ) -> InboundEventRecord | None:
        params: list[Any] = [event_id]
        sql = "SELECT * FROM inbound_events WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._event_from_row(row)

    def get_event_by_provider_event_id(
        self,
        integration_id: str,
        provider_event_id: str,
        *,
        account_id: str | None = None,
    ) -> InboundEventRecord | None:
        params: list[Any] = [integration_id, provider_event_id]
        sql = """
            SELECT *
            FROM inbound_events
            WHERE integration_id = ? AND provider_event_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._event_from_row(row)

    def get_event_by_provider_message_id(
        self,
        integration_id: str,
        provider_message_id: str,
        *,
        account_id: str | None = None,
    ) -> InboundEventRecord | None:
        params: list[Any] = [integration_id, provider_message_id]
        sql = """
            SELECT *
            FROM inbound_events
            WHERE integration_id = ? AND provider_message_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += """
            ORDER BY received_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._event_from_row(row)

    def get_event_by_dedup_key(
        self,
        integration_id: str,
        dedup_key: str,
        *,
        account_id: str | None = None,
    ) -> InboundEventRecord | None:
        params: list[Any] = [integration_id, dedup_key]
        sql = """
            SELECT *
            FROM inbound_events
            WHERE integration_id = ? AND dedup_key = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._event_from_row(row)

    def list_events(
        self,
        *,
        account_id: str | None = None,
        integration_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[InboundEventRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if integration_id is not None:
            clauses.append("integration_id = ?")
            params.append(integration_id)
        if status is not None:
            clauses.append("normalized_status = ?")
            params.append(status)

        sql = "SELECT * FROM inbound_events"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY received_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._event_from_row(row) for row in rows]

    def update_event(self, record: InboundEventRecord) -> InboundEventRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            updated = conn.execute(
                """
                UPDATE inbound_events
                SET integration_id = ?, account_id = ?, provider_event_id = ?, provider_message_id = ?, provider_chat_id = ?,
                    provider_thread_id = ?, provider_user_id = ?, event_type = ?, received_at = ?,
                    signature_valid = ?, dedup_key = ?, raw_headers_json = ?, raw_headers_size_bytes = ?,
                    raw_headers_truncated = ?, raw_headers_redacted_fields_json = ?, raw_payload_json = ?,
                    raw_payload_size_bytes = ?, raw_payload_truncated = ?, raw_payload_redacted_fields_json = ?,
                    normalized_status = ?, normalized_error = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    record.integration_id,
                    resolved_account_id,
                    record.provider_event_id or None,
                    record.provider_message_id or None,
                    record.provider_chat_id,
                    record.provider_thread_id,
                    record.provider_user_id,
                    record.event_type,
                    record.received_at,
                    int(record.signature_valid),
                    record.dedup_key,
                    self._json_dump(record.raw_headers),
                    record.raw_headers_size_bytes,
                    int(record.raw_headers_truncated),
                    self._json_dump(record.raw_headers_redacted_fields),
                    self._json_dump(record.raw_payload),
                    record.raw_payload_size_bytes,
                    int(record.raw_payload_truncated),
                    self._json_dump(record.raw_payload_redacted_fields),
                    record.normalized_status,
                    record.normalized_error,
                    self._json_dump(record.metadata),
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Inbound event not found: {record.id}")
        return record.model_copy(update={"account_id": resolved_account_id})


class ConversationBindingStore(_IntegrationStoreBase):
    """管理渠道到会话的绑定关系。"""

    def _binding_from_row(self, row: sqlite3.Row) -> ConversationBindingRecord:
        return ConversationBindingRecord(
            id=row["id"],
            integration_id=row["integration_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            tenant_id=row["tenant_id"],
            chat_id=row["chat_id"],
            thread_id=row["thread_id"],
            binding_scope=row["binding_scope"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            enabled=bool(row["enabled"]),
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_message_at=row["last_message_at"],
        )

    def create_binding(self, record: ConversationBindingRecord) -> ConversationBindingRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            conn.execute(
                """
                INSERT INTO conversation_bindings (
                    id, integration_id, account_id, tenant_id, chat_id, thread_id, binding_scope, agent_id,
                    session_id, enabled, metadata_json, created_at, updated_at, last_message_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.integration_id,
                    resolved_account_id,
                    record.tenant_id,
                    record.chat_id,
                    record.thread_id,
                    record.binding_scope,
                    record.agent_id,
                    record.session_id,
                    int(record.enabled),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.last_message_at,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def get_binding(
        self,
        binding_id: str,
        *,
        account_id: str | None = None,
    ) -> ConversationBindingRecord | None:
        params: list[Any] = [binding_id]
        sql = "SELECT * FROM conversation_bindings WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._binding_from_row(row)

    def find_binding(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        tenant_id: str = "",
        chat_id: str = "",
        thread_id: str = "",
        binding_scope: str | None = None,
        agent_id: str | None = None,
        enabled: bool | None = True,
    ) -> ConversationBindingRecord | None:
        clauses = [
            "integration_id = ?",
            "tenant_id = ?",
            "chat_id = ?",
            "thread_id = ?",
        ]
        params: list[Any] = [integration_id, tenant_id, chat_id, thread_id]
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if binding_scope is not None:
            clauses.append("binding_scope = ?")
            params.append(binding_scope)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(int(enabled))

        sql = f"""
            SELECT *
            FROM conversation_bindings
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(last_message_at, updated_at) DESC, updated_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._binding_from_row(row)

    def list_bindings(
        self,
        *,
        account_id: str | None = None,
        integration_id: str | None = None,
        tenant_id: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
        binding_scope: str | None = None,
        agent_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[ConversationBindingRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if integration_id is not None:
            clauses.append("integration_id = ?")
            params.append(integration_id)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if binding_scope is not None:
            clauses.append("binding_scope = ?")
            params.append(binding_scope)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(int(enabled))

        sql = "SELECT * FROM conversation_bindings"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY COALESCE(last_message_at, updated_at) DESC, updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._binding_from_row(row) for row in rows]

    def update_binding(self, record: ConversationBindingRecord) -> ConversationBindingRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            updated = conn.execute(
                """
                UPDATE conversation_bindings
                SET integration_id = ?, account_id = ?, tenant_id = ?, chat_id = ?, thread_id = ?, binding_scope = ?,
                    agent_id = ?, session_id = ?, enabled = ?, metadata_json = ?, created_at = ?,
                    updated_at = ?, last_message_at = ?
                WHERE id = ?
                """,
                (
                    record.integration_id,
                    resolved_account_id,
                    record.tenant_id,
                    record.chat_id,
                    record.thread_id,
                    record.binding_scope,
                    record.agent_id,
                    record.session_id,
                    int(record.enabled),
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.last_message_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Conversation binding not found: {record.id}")
        return record.model_copy(update={"account_id": resolved_account_id})

    def delete_binding(self, binding_id: str, *, account_id: str | None = None) -> bool:
        params: list[Any] = [binding_id]
        sql = "DELETE FROM conversation_bindings WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0


class DeliveryStore(_IntegrationStoreBase):
    """管理出站投递与重试尝试明细。"""

    def _delivery_from_row(self, row: sqlite3.Row) -> OutboundDeliveryRecord:
        return OutboundDeliveryRecord(
            id=row["id"],
            integration_id=row["integration_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            run_id=row["run_id"],
            session_id=row["session_id"],
            inbound_event_id=row["inbound_event_id"],
            provider_chat_id=row["provider_chat_id"],
            provider_thread_id=row["provider_thread_id"],
            provider_message_id=row["provider_message_id"],
            delivery_type=row["delivery_type"],
            payload=self._json_load(row["payload_json"], {}),
            status=row["status"],
            attempt_count=row["attempt_count"],
            last_attempt_at=row["last_attempt_at"],
            error_summary=row["error_summary"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _attempt_from_row(self, row: sqlite3.Row) -> DeliveryAttemptRecord:
        return DeliveryAttemptRecord(
            id=row["id"],
            delivery_id=row["delivery_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            attempt_number=row["attempt_number"],
            status=row["status"],
            request_payload=self._json_load(row["request_payload_json"], {}),
            response_payload=self._json_load(row["response_payload_json"], {}),
            error_summary=row["error_summary"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def create_delivery(self, record: OutboundDeliveryRecord) -> OutboundDeliveryRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            conn.execute(
                """
                INSERT INTO outbound_deliveries (
                    id, integration_id, account_id, run_id, session_id, inbound_event_id, provider_chat_id,
                    provider_thread_id, provider_message_id, delivery_type, payload_json, status,
                    attempt_count, last_attempt_at, error_summary, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.integration_id,
                    resolved_account_id,
                    record.run_id,
                    record.session_id,
                    record.inbound_event_id,
                    record.provider_chat_id,
                    record.provider_thread_id,
                    record.provider_message_id,
                    record.delivery_type,
                    self._json_dump(record.payload),
                    record.status,
                    record.attempt_count,
                    record.last_attempt_at,
                    record.error_summary,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def get_delivery(
        self,
        delivery_id: str,
        *,
        account_id: str | None = None,
    ) -> OutboundDeliveryRecord | None:
        params: list[Any] = [delivery_id]
        sql = "SELECT * FROM outbound_deliveries WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._delivery_from_row(row)

    def list_deliveries(
        self,
        *,
        account_id: str | None = None,
        integration_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[OutboundDeliveryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if integration_id is not None:
            clauses.append("integration_id = ?")
            params.append(integration_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM outbound_deliveries"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._delivery_from_row(row) for row in rows]

    def update_delivery(self, record: OutboundDeliveryRecord) -> OutboundDeliveryRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "integrations",
                "id",
                record.integration_id,
                fallback=record.account_id,
            )
            updated = conn.execute(
                """
                UPDATE outbound_deliveries
                SET integration_id = ?, account_id = ?, run_id = ?, session_id = ?, inbound_event_id = ?, provider_chat_id = ?,
                    provider_thread_id = ?, provider_message_id = ?, delivery_type = ?, payload_json = ?,
                    status = ?, attempt_count = ?, last_attempt_at = ?, error_summary = ?, metadata_json = ?,
                    created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.integration_id,
                    resolved_account_id,
                    record.run_id,
                    record.session_id,
                    record.inbound_event_id,
                    record.provider_chat_id,
                    record.provider_thread_id,
                    record.provider_message_id,
                    record.delivery_type,
                    self._json_dump(record.payload),
                    record.status,
                    record.attempt_count,
                    record.last_attempt_at,
                    record.error_summary,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Outbound delivery not found: {record.id}")
        return record.model_copy(update={"account_id": resolved_account_id})

    def create_attempt(self, record: DeliveryAttemptRecord) -> DeliveryAttemptRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_account_id(
                conn,
                "outbound_deliveries",
                "id",
                record.delivery_id,
                fallback=record.account_id,
            )
            conn.execute(
                """
                INSERT INTO delivery_attempts (
                    id, delivery_id, account_id, attempt_number, status, request_payload_json,
                    response_payload_json, error_summary, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.delivery_id,
                    resolved_account_id,
                    record.attempt_number,
                    record.status,
                    self._json_dump(record.request_payload),
                    self._json_dump(record.response_payload),
                    record.error_summary,
                    record.started_at,
                    record.finished_at,
                ),
            )
            conn.execute(
                """
                UPDATE outbound_deliveries
                SET attempt_count = CASE
                        WHEN attempt_count < ? THEN ?
                        ELSE attempt_count
                    END,
                    last_attempt_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    record.attempt_number,
                    record.attempt_number,
                    record.started_at,
                    record.finished_at or record.started_at,
                    record.delivery_id,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def list_attempts(
        self,
        delivery_id: str,
        *,
        account_id: str | None = None,
    ) -> list[DeliveryAttemptRecord]:
        params: list[Any] = [delivery_id]
        sql = """
            SELECT *
            FROM delivery_attempts
            WHERE delivery_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY attempt_number ASC"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                tuple(params),
            ).fetchall()
        return [self._attempt_from_row(row) for row in rows]
