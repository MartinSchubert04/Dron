import asyncio
import json
import logging
import os
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)

_UDP_PORT = int(os.getenv("ESP32_UDP_PORT", "4210"))
_DISCONNECT_TIMEOUT = 3.0  # seconds without a packet → mark disconnected


class Esp32TelemetryService:
    """Receives telemetry pushed by the ESP32-C3 via UDP broadcast (port 4210).

    The ESP32 broadcasts a JSON packet at ~20 Hz; this service listens passively
    and updates _latest on each received packet. No polling, no TCP overhead.
    """

    def __init__(self) -> None:
        self._latest: dict = {
            "roll": 0.0, "pitch": 0.0, "imu_ok": False, "connected": False,
            "gps_fix": False, "satellites": 0,
            "lat": 0.0, "lng": 0.0, "altitude_m": 0.0, "speed_kmh": 0.0,
        }
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_packet: float = 0.0
        # Kept for backward-compat with /settings/esp32_url API; not used for transport
        self._url: str = os.getenv("ESP32_TELEMETRY_URL", "").strip()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_url(self, url: str) -> None:
        self._url = url.strip()

    def get_url(self) -> str:
        return self._url

    def get_latest(self) -> dict:
        return dict(self._latest)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._listen_loop())
        logger.info("[esp32] UDP listener started on 0.0.0.0:%d", _UDP_PORT)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _listen_loop(self) -> None:
        loop = asyncio.get_event_loop()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", _UDP_PORT))
        sock.setblocking(False)
        logger.info("[esp32] listening on 0.0.0.0:%d", _UDP_PORT)

        try:
            while self._running:
                try:
                    data, _addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 1024),
                        timeout=_DISCONNECT_TIMEOUT,
                    )
                    self._handle_packet(data)
                except asyncio.TimeoutError:
                    if self._last_packet > 0:
                        self._latest["connected"] = False
                        logger.debug("[esp32] no packet for %.1fs → disconnected", _DISCONNECT_TIMEOUT)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.debug("[esp32] recv error: %s", exc)
                    await asyncio.sleep(0.05)
        finally:
            sock.close()

    def _handle_packet(self, data: bytes) -> None:
        try:
            j = json.loads(data)
            self._latest = {
                "roll":       float(j.get("roll",       0.0)),
                "pitch":      float(j.get("pitch",      0.0)),
                "imu_ok":     bool(j.get("imu_ok",     False)),
                "connected":  True,
                "gps_fix":    bool(j.get("gps_fix",    False)),
                "satellites": int(j.get("satellites",   0)),
                "lat":        float(j.get("lat",        0.0)),
                "lng":        float(j.get("lng",        0.0)),
                "altitude_m": float(j.get("altitude_m", 0.0)),
                "speed_kmh":  float(j.get("speed_kmh",  0.0)),
            }
            self._last_packet = time.monotonic()
        except Exception as exc:
            logger.debug("[esp32] packet parse error: %s", exc)
