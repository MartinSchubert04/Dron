import { useCallback, useEffect, useRef, useState } from "react";
import { useSettings } from "../context/SettingsContext";

export interface TelemetryData {
  roll:      number;
  pitch:     number;
  imu_ok:    boolean;
  connected: boolean;
}

const EMPTY: TelemetryData = { roll: 0, pitch: 0, imu_ok: false, connected: false };

export function useTelemetry(pollHz = 10) {
  const { apiBase } = useSettings();
  const [data, setData]       = useState<TelemetryData>(EMPTY);
  const [esp32Url, setEsp32UrlState] = useState<string>("");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/telemetry`, { signal: AbortSignal.timeout(800) });
      if (!res.ok) return;
      const json = await res.json();
      setData({
        roll:      Number(json.roll  ?? 0),
        pitch:     Number(json.pitch ?? 0),
        imu_ok:    Boolean(json.imu_ok),
        connected: Boolean(json.connected),
      });
    } catch {
      setData(prev => ({ ...prev, connected: false }));
    }
  }, [apiBase]);

  useEffect(() => {
    poll();
    intervalRef.current = setInterval(poll, 1000 / pollHz);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [poll, pollHz]);

  // Load persisted esp32Url from backend on mount
  useEffect(() => {
    fetch(`${apiBase}/settings/esp32_url`)
      .then(r => r.json())
      .then(j => setEsp32UrlState(j.url ?? ""))
      .catch(() => {});
  }, [apiBase]);

  const setEsp32Url = useCallback(async (url: string) => {
    setEsp32UrlState(url);
    try {
      await fetch(`${apiBase}/settings/esp32_url?url=${encodeURIComponent(url)}`, { method: "POST" });
    } catch { /* backend might be down */ }
  }, [apiBase]);

  return { data, esp32Url, setEsp32Url };
}
