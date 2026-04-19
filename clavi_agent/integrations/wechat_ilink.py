"""Native WeChat iLink client helpers used by the WeChat integration."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import secrets
from typing import Any

import httpx
from pydantic import BaseModel, Field

DEFAULT_ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
QR_CODE_URL = f"{DEFAULT_ILINK_BASE_URL}/ilink/bot/get_bot_qrcode?bot_type=3"
QR_STATUS_URL = f"{DEFAULT_ILINK_BASE_URL}/ilink/bot/get_qrcode_status?qrcode="
STATUS_WAIT = "wait"
STATUS_SCANNED = "scaned"
STATUS_CONFIRMED = "confirmed"
STATUS_EXPIRED = "expired"


class WeChatLoginQRCode(BaseModel):
    qrcode: str
    qr_content: str


class WeChatILinkCredentials(BaseModel):
    bot_token: str
    ilink_bot_id: str
    base_url: str = DEFAULT_ILINK_BASE_URL
    ilink_user_id: str = ""


class WeChatILinkClient:
    """Async HTTP client for the undocumented WeChat iLink API."""

    def __init__(
        self,
        credentials: WeChatILinkCredentials,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._credentials = credentials
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._wechat_uin = _generate_wechat_uin()

    @property
    def bot_id(self) -> str:
        return self._credentials.ilink_bot_id

    async def get_updates(self, cursor: str = "") -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/getupdates",
            {
                "get_updates_buf": cursor,
                "base_info": {"channel_version": "1.0.0"},
            },
            timeout_seconds=40.0,
        )

    async def send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/sendmessage",
            payload,
        )

    async def get_config(self, user_id: str, context_token: str = "") -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/getconfig",
            {
                "ilink_user_id": user_id,
                "context_token": context_token,
                "base_info": {},
            },
            timeout_seconds=10.0,
        )

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        async with httpx.AsyncClient(
            base_url=self._credentials.base_url or DEFAULT_ILINK_BASE_URL,
            timeout=timeout,
            transport=self._transport,
        ) as client:
            response = await client.post(
                path,
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=self._headers(),
            )
        response.raise_for_status()
        return _safe_json_dict(response)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self._credentials.bot_token}",
            "X-WECHAT-UIN": self._wechat_uin,
        }


async def fetch_login_qr_code(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> WeChatLoginQRCode:
    """Request one QR code from the WeChat iLink login endpoint."""

    async with httpx.AsyncClient(timeout=40.0, transport=transport) as client:
        response = await client.get(QR_CODE_URL)
    response.raise_for_status()
    payload = _safe_json_dict(response)

    qrcode = str(payload.get("qrcode") or "").strip()
    qr_content = str(payload.get("qrcode_img_content") or "").strip()
    if not qrcode or not qr_content:
        raise RuntimeError("WeChat iLink did not return a usable QR code payload.")
    return WeChatLoginQRCode(qrcode=qrcode, qr_content=qr_content)


async def poll_login_status(
    qrcode: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    on_status=None,
) -> WeChatILinkCredentials:
    """Poll the WeChat iLink QR login status until confirmed or expired."""

    normalized_qrcode = str(qrcode or "").strip()
    if not normalized_qrcode:
        raise ValueError("qrcode is required.")

    status_url = f"{QR_STATUS_URL}{normalized_qrcode}"
    last_status = ""
    async with httpx.AsyncClient(timeout=40.0, transport=transport) as client:
        while True:
            try:
                response = await client.get(status_url)
                response.raise_for_status()
            except httpx.TimeoutException:
                continue

            payload = _safe_json_dict(response)
            status = str(payload.get("status") or "").strip().lower()
            if status and status != last_status and on_status is not None:
                maybe_awaitable = on_status(status)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            last_status = status or last_status

            if status == STATUS_CONFIRMED:
                credentials = WeChatILinkCredentials(
                    bot_token=str(payload.get("bot_token") or "").strip(),
                    ilink_bot_id=str(payload.get("ilink_bot_id") or "").strip(),
                    base_url=str(payload.get("baseurl") or DEFAULT_ILINK_BASE_URL).strip()
                    or DEFAULT_ILINK_BASE_URL,
                    ilink_user_id=str(payload.get("ilink_user_id") or "").strip(),
                )
                if not credentials.bot_token or not credentials.ilink_bot_id:
                    raise RuntimeError("WeChat iLink login succeeded without usable credentials.")
                return credentials
            if status == STATUS_EXPIRED:
                raise RuntimeError("The WeChat QR code expired before login was confirmed.")
            if status in {STATUS_WAIT, STATUS_SCANNED, ""}:
                await asyncio.sleep(1.0)
                continue
            await asyncio.sleep(0.2)


def _generate_wechat_uin() -> str:
    raw = str(secrets.randbelow(2**32 - 1)).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _safe_json_dict(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("WeChat iLink returned non-JSON data.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("WeChat iLink returned an unexpected JSON payload.")
    return payload
