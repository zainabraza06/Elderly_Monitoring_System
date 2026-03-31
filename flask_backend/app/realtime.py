from __future__ import annotations

import asyncio

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder

from .schemas import MonitorEvent


class MonitorConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, event: MonitorEvent) -> None:
        message = jsonable_encoder(event)
        stale_connections: list[WebSocket] = []

        async with self._lock:
            connections = list(self._connections)

        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                stale_connections.append(websocket)

        if stale_connections:
            async with self._lock:
                for websocket in stale_connections:
                    self._connections.discard(websocket)
