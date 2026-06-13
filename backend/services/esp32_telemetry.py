import asyncio
import http.client
import json
import logging
import os
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1    # 10 Hz target
_REQUEST_TIMEOUT = 0.4  # ESP32 WiFi RTT should be <50ms; 400ms gives margin


class Esp32TelemetryService:
    """Polls an ESP32-C3 IMU endpoint and caches the latest reading."""

    def __init__(self) -> None:
        raw = os.getenv("ESP32_TELEMETRY_URL", "").strip()
        self._url: str = raw
        self._latest: dict = {
            "roll": 0.0, "pitch": 0.0, "imu_ok": False, "esp32_ok": False,
            "gps_ok": False, "satellites": 0,
            "lat": 0.0, "lng": 0.0, "altitude_m": 0.0, "speed_kmh": 0.0,
        }
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._conn: Optional[http.client.HTTPConnection] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def set_url(self, url: str) -> None:
        if url.strip() != self._url:
            self._close_conn()
        self._url = url.strip()
        logger.info("[esp32] telemetry URL → %s", self._url or "(cleared)")

    def get_url(self) -> str:
        return self._url

    def get_latest(self) -> dict:
        return dict(self._latest)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._poll_loop())
        logger.info("[esp32] telemetry poller started (url=%s)", self._url or "not set")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._close_conn()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _close_conn(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    async def _poll_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            if not self._url:
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            deadline = loop.time() + _POLL_INTERVAL

            try:
                data = await loop.run_in_executor(None, self._fetch)
                if data is not None:
                    self._latest = {
                        "roll":       float(data.get("roll", 0.0)),
                        "pitch":      float(data.get("pitch", 0.0)),
                        "imu_ok":     bool(data.get("imu_ok", True)),
                        "esp32_ok":   True,
                        "gps_ok":     bool(data.get("gps_ok", False)),
                        "satellites": int(data.get("satellites", 0)),
                        "lat":        float(data.get("lat", 0.0)),
                        "lng":        float(data.get("lng", 0.0)),
                        "altitude_m": float(data.get("altitude_m", 0.0)),
                        "speed_kmh":  float(data.get("speed_kmh", 0.0)),
                    }
                else:
                    self._latest["esp32_ok"] = False
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._latest["esp32_ok"] = False
                logger.debug("[esp32] poll error: %s", exc)

            remaining = deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

    def _fetch(self) -> Optional[dict]:
        parsed = urllib.parse.urlparse(self._url)
        host   = parsed.hostname or ""
        port   = parsed.port or 80
        path   = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")

        try:
            if self._conn is None:
                self._conn = http.client.HTTPConnection(host, port, timeout=_REQUEST_TIMEOUT)

            self._conn.request("GET", path)
            resp = self._conn.getresponse()
            body = resp.read()
            if resp.will_close:
                self._close_conn()
            return json.loads(body)

        except Exception:
            self._close_conn()
            try:
                self._conn = http.client.HTTPConnection(host, port, timeout=_REQUEST_TIMEOUT)
                self._conn.request("GET", path)
                resp = self._conn.getresponse()
                body = resp.read()
                if resp.will_close:
                    self._close_conn()
                return json.loads(body)
            except Exception:
                self._close_conn()
                return None
