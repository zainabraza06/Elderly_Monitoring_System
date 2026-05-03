"""In-process WebSocket fan-out for caregiver clients (alert push)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.websockets import WebSocket


class CaregiverRealtimeHub:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def register(self, caregiver_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._rooms.setdefault(caregiver_id, set()).add(ws)

    async def unregister(self, caregiver_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._rooms.get(caregiver_id)
            if room and ws in room:
                room.discard(ws)
                if not room:
                    self._rooms.pop(caregiver_id, None)

    async def broadcast_to_caregiver(self, caregiver_id: str, payload: dict[str, Any]) -> None:
        msg = json.dumps({"type": "alert", "data": payload})
        async with self._lock:
            clients = list(self._rooms.get(caregiver_id, set()))
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                await self.unregister(caregiver_id, ws)


hub = CaregiverRealtimeHub()
