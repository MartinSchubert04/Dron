import asyncio
import json
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 0.8   # seconds — fast enough to not block 10 Hz polling


class Esp32TelemetryService:
    """Polls an ESP32-C3 IMU endpoint and caches the latest reading."""

    def __init__(self) -> None:
        raw = os.getenv("ESP32_TELEMETRY_URL", "").strip()
        self._url: str = raw
        self._latest: dict = {"roll": 0.0, "pitch": 0.0, "imu_ok": False, "connected": False}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────────

    def set_url(self, url: str) -> None:
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

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            if self._url:
                try:
                    data = await asyncio.get_event_loop().run_in_executor(None, self._fetch)
                    if data is not None:
                        self._latest = {
                            "roll":      float(data.get("roll", 0.0)),
                            "pitch":     float(data.get("pitch", 0.0)),
                            "imu_ok":    bool(data.get("imu_ok", True)),
                            "connected": True,
                        }
                    else:
                        self._latest["connected"] = False
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._latest["connected"] = False
                    logger.debug("[esp32] poll error: %s", exc)
            await asyncio.sleep(0.1)   # 10 Hz

    def _fetch(self) -> Optional[dict]:
        try:
            with urllib.request.urlopen(self._url, timeout=_DEFAULT_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception:
            return None
